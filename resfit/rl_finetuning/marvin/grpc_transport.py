from __future__ import annotations

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
try:
    from lerobot.common.transport import services_pb2, services_pb2_grpc
    from lerobot.common.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks
    from lerobot.scripts.rl import learner_service
except (ModuleNotFoundError, SyntaxError):
    # Current LeRobot moved transport and RL modules without changing their
    # wire protocol. Keep the original residual gRPC implementation usable.
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks

    # lerobot-vlahost's ``lerobot.rl`` package uses Python 3.12 generic
    # syntax in its top-level __init__.py.  OpenPI currently runs Python 3.11,
    # so importing that package just to get this tiny wire service fails.  The
    # service below is deliberately kept local and uses the stable protobuf
    # contract only (Ready, StreamParameters, SendTransitions).
    class _LearnerService(services_pb2_grpc.LearnerServiceServicer):
        MAX_MESSAGE_SIZE = 4 * 1024 * 1024
        MAX_WORKERS = 3

        def __init__(self, *, shutdown_event, parameters_queue,
                     seconds_between_pushes, transition_queue,
                     interaction_message_queue):
            self.shutdown_event = shutdown_event
            self.parameters_queue = parameters_queue
            self.seconds_between_pushes = seconds_between_pushes
            self.transition_queue = transition_queue
            self.interaction_message_queue = interaction_message_queue

        def Ready(self, request, context):  # noqa: N802
            return services_pb2.Empty()

        def StreamParameters(self, request, context):  # noqa: N802
            last_push = 0.0
            while not self.shutdown_event.is_set():
                wait = self.seconds_between_pushes - (time.time() - last_push)
                if wait > 0:
                    self.shutdown_event.wait(wait)
                    continue
                try:
                    payload = self.parameters_queue.get(timeout=0.001)
                except queue.Empty:
                    self.shutdown_event.wait(self.seconds_between_pushes)
                    continue
                yield from send_bytes_in_chunks(
                    payload, services_pb2.Parameters,
                    log_prefix="[LEARNER] parameters", silent=True
                )
                last_push = time.time()

        def SendTransitions(self, request_iterator, context):  # noqa: N802
            receive_bytes_in_chunks(
                request_iterator, self.transition_queue, self.shutdown_event,
                log_prefix="[LEARNER] transitions"
            )
            return services_pb2.Empty()

        def SendInteractions(self, request_iterator, context):  # noqa: N802
            receive_bytes_in_chunks(
                request_iterator, self.interaction_message_queue,
                self.shutdown_event, log_prefix="[LEARNER] interactions"
            )
            return services_pb2.Empty()

    class _LearnerServiceModule:
        LearnerService = _LearnerService
        MAX_MESSAGE_SIZE = _LearnerService.MAX_MESSAGE_SIZE
        MAX_WORKERS = _LearnerService.MAX_WORKERS

    learner_service = _LearnerServiceModule()


def start_learner_server(
    *, host: str, port: int, parameters_queue, transition_queue, shutdown_event, push_frequency_s: float
):
    service = learner_service.LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=push_frequency_s,
        transition_queue=transition_queue,
        interaction_message_queue=queue.Queue(),
    )
    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", learner_service.MAX_MESSAGE_SIZE),
        ],
    )
    services_pb2_grpc.add_LearnerServiceServicer_to_server(service, server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"Could not bind Marvin learner gRPC server to {host}:{port}")
    server.start()
    logging.warning("Marvin learner gRPC listening on %s:%s", host, port)
    return server


def actor_stub(host: str, port: int):
    service_config = json.dumps({
        "methodConfig": [{
            "name": [{}],
            "retryPolicy": {
                "maxAttempts": 5,
                "initialBackoff": "0.1s",
                "maxBackoff": "2s",
                "backoffMultiplier": 2,
                "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
            },
        }]
    })
    channel = grpc.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_receive_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.enable_retries", 1),
            ("grpc.service_config", service_config),
        ],
    )
    return services_pb2_grpc.LearnerServiceStub(channel), channel


def wait_until_ready(stub, shutdown_event: threading.Event) -> bool:
    while not shutdown_event.is_set():
        try:
            stub.Ready(services_pb2.Empty(), timeout=2)
            return True
        except grpc.RpcError:
            shutdown_event.wait(1)
    return False


def receive_parameters(stub, output_queue, shutdown_event: threading.Event) -> None:
    while not shutdown_event.is_set():
        try:
            if not wait_until_ready(stub, shutdown_event):
                return
            receive_bytes_in_chunks(
                stub.StreamParameters(services_pb2.Empty()), output_queue, shutdown_event,
                log_prefix="[MARVIN ACTOR] parameters",
            )
        except grpc.RpcError as exc:
            logging.warning("Marvin parameter stream disconnected: %s", exc)
            shutdown_event.wait(1)


def send_transition_batches(stub, input_queue, shutdown_event: threading.Event) -> None:
    while not shutdown_event.is_set():
        try:
            payload = input_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        while not shutdown_event.is_set():
            try:
                if not wait_until_ready(stub, shutdown_event):
                    return
                stub.SendTransitions(
                    send_bytes_in_chunks(payload, services_pb2.Transition, "[MARVIN ACTOR] transitions"),
                    timeout=120,
                )
                break
            except grpc.RpcError as exc:
                logging.warning("Marvin transition send failed; retrying: %s", exc)
                time.sleep(1)

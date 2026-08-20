from __future__ import annotations

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from lerobot.common.transport import services_pb2, services_pb2_grpc
from lerobot.common.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks
from lerobot.scripts.rl import learner_service


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

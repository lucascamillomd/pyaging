"""``download`` against a local HTTP server: completion, report hook and stalls."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock

import pytest

import pyaging.utils as utils

PAYLOAD = b"x" * 20_000


class _PayloadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, *args):
        pass


@pytest.fixture
def payload_url():
    server = HTTPServer(("127.0.0.1", 0), _PayloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/data/file.pt"
    server.shutdown()
    server.server_close()


@pytest.fixture
def stalled_url():
    """A server that accepts the connection and never answers."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    connections = []
    threading.Thread(target=lambda: connections.append(listener.accept()), daemon=True).start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}/file.pt"
    for connection, _ in connections:
        connection.close()
    listener.close()


def test_download_writes_complete_file_and_reports_progress(payload_url, tmp_path):
    logger = Mock()

    utils.download(payload_url, str(tmp_path), logger)

    assert (tmp_path / "file.pt").read_bytes() == PAYLOAD
    assert [path.name for path in tmp_path.iterdir()] == ["file.pt"]
    hook_calls = logger.request_report_hook.call_args_list
    assert hook_calls[0].args == (0, 8192, len(PAYLOAD))
    assert hook_calls[-1].args[0] * 8192 >= len(PAYLOAD)


def test_stalled_download_times_out_and_leaves_no_temporary_files(stalled_url, tmp_path):
    outcome = []

    def run():
        try:
            utils.download(stalled_url, str(tmp_path), Mock(), timeout=0.2)
        except Exception as error:  # noqa: BLE001 - the test inspects whatever surfaced
            outcome.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(10)

    assert not worker.is_alive(), "download blocked instead of timing out"
    assert len(outcome) == 1
    assert isinstance(outcome[0], (TimeoutError, OSError))
    assert list(tmp_path.iterdir()) == []


def test_download_has_a_default_socket_timeout():
    import inspect

    assert inspect.signature(utils.download).parameters["timeout"].default == 60

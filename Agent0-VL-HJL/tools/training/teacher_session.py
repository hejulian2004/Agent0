"""Reuse a matching teacher or own one subprocess for explicit generation."""
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse
from urllib.request import urlopen

from tools.training.teacher_profile import verify


@contextmanager
def teacher_session(root, config):
    teacher = config["sft_data_generation"]["local_teacher"]
    parsed = urlparse(teacher["base_url"])
    if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        raise ValueError("Local teacher must be on this host")
    process, log = None, None
    try:
        try:
            connection = socket.create_connection((parsed.hostname, parsed.port or 8000), timeout=2)
        except ConnectionRefusedError:
            connection = None
        if connection:
            connection.close()
            verify(config)
        else:
            from tools.local_workflows import teacher_command
            directory = root / "logs/teacher" / datetime.now().strftime("teacher_%Y%m%d_%H%M%S_%f")
            directory.mkdir(parents=True)
            (directory / "README.md").write_text("Owned teacher for explicit balanced generation. BF16, FP8 KV, TP4 PP1, MTP1; context57344, sequences8, batched16384, memory0.90. Started by tools/training/teacher_session.py using config.yaml/local_4090.\n")
            log = (directory / "server.log").open("ab")
            process = subprocess.Popen(teacher_command(root, config), cwd=root, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Owned teacher exited during startup; inspect " + str(directory))
                try:
                    with urlopen(f"{parsed.scheme}://{parsed.netloc}/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Teacher startup timed out")
            verify(config)
        yield
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if log:
            log.close()

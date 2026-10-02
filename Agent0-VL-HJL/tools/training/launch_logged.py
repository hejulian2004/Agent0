"""Log/terminal wrapper that signals only its own launched process group."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import fcntl


def run(command, log_path, env):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    lock = (Path(log_path).parent / '.run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ValueError('Run is still active; refusing a second launch/resume') from exc
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    renderer = subprocess.Popen([sys.executable, "-m", "tools.training.terminal_progress", str(log_path)],
                                stdin=process.stdout, env=env)
    process.stdout.close()
    try:
        result = process.wait()
        renderer.wait()
        return result
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        renderer.wait()
        return 130
    finally:
        lock.close()

"""Mirror raw logs to disk and reserve a terminal footer for RL progress."""

import codecs
import os
from pathlib import Path
import re
import select
import shutil
import signal
import sys
import subprocess
import threading


ANSI = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')



class ActorMemorySampler:
    """Physical GPU memory snapshots during Actor forward/backward only."""

    def __init__(self):
        self.active = threading.Event()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.peaks = {}
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop.wait(1):
            if not self.active.is_set():
                continue
            command = ['nvidia-smi', '--query-gpu=index,memory.used',
                       '--format=csv,noheader,nounits']
            devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if devices:
                command += ['-i', devices]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=2)
                if result.returncode != 0:
                    continue
                values = {}
                for line in result.stdout.splitlines():
                    gpu, used = line.split(',')
                    values[gpu.strip()] = float(used.strip()) / 1024
                with self.lock:
                    for gpu, used in values.items():
                        self.peaks[gpu] = max(self.peaks.get(gpu, 0), used)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                continue

    def text(self):
        with self.lock:
            return ' '.join(f'GPU{gpu}={peak:.2f}' for gpu, peak in sorted(self.peaks.items())) or 'N/A'

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


class TerminalProgress:
    def __init__(self, stream, memory):
        self.stream = stream
        self.memory = memory
        self.footer_rows = 5
        self.enabled = stream.isatty() and os.environ.get('TERM', '') != 'dumb'
        self.size = None
        self.progress = 'Training Progress: initializing'
        self.stage = ''
        self.step_timing = 'Step timing: N/A (waiting for completed step)'
        self.trajectories = ''
        self.tool_success = 'N/A (waiting for completed step)'
        self.answer_accuracy = 'N/A (waiting for completed step)'

    def resize(self):
        if not self.enabled:
            return
        size = shutil.get_terminal_size(fallback=(100, 24))
        if size.lines < self.footer_rows + 2:
            self.restore()
            self.enabled = False
            return
        if size != self.size:
            self.size = size
            self.stream.write(f'\x1b[r\x1b[1;{size.lines-self.footer_rows}r\x1b[{size.lines-self.footer_rows};1H')
            self.stream.flush()

    def draw(self):
        if not self.enabled:
            return
        self.resize()
        if not self.enabled:
            return
        status = ' | '.join(x for x in (self.progress, self.stage, self.trajectories) if x)
        rows = (status, f'Global tool success: {self.tool_success}',
                f'Global answer accuracy: {self.answer_accuracy}',
                f'Actor peak (sampled GiB): {self.memory.text()}', self.step_timing)
        self.stream.write('\x1b7')
        for offset, value in enumerate(rows):
            value = value[:max(1, self.size.columns-1)]
            self.stream.write(f'\x1b[{self.size.lines-self.footer_rows+1+offset};1H\x1b[2K{value}')
        self.stream.write('\x1b8')
        self.stream.flush()

    def observe(self, text):
        clean = ANSI.sub('', text).strip()
        tool = re.search(r'overall_tool_success=([^\s]+)', clean)
        answer = re.search(r'overall_answer_accuracy=([^\s]+)', clean)
        if tool:
            self.tool_success = tool.group(1)
        if answer:
            self.answer_accuracy = answer.group(1)
        if '[Agent0-VL trainer] actor_update_start' in clean:
            self.memory.active.set()
        elif '[Agent0-VL trainer] actor_update_done' in clean:
            self.memory.active.clear()
        if 'timing_s/gen:' in clean and 'timing_s/update_actor:' in clean:
            metrics = dict(re.findall(r'([\w/]+):(-?\d+(?:\.\d+)?)', clean))
            step = metrics.get('step', '?')
            parts = [f'Step {step} timing']
            for key, label in (('timing_s/gen', 'generation'),
                               ('timing_s/update_actor', 'Actor'),
                               ('timing_s/step', 'total')):
                if key in metrics:
                    parts.append(f'{label}={float(metrics[key]):.1f}s')
            self.step_timing = ' | '.join(parts)
        self.clean = clean

    def line(self, text):
        self.observe(text)
        clean = self.clean
        progress = re.search(r'Training Progress:.*', clean)
        if progress:
            self.progress = progress.group()
        else:
            stage = re.search(r'\[Agent0-VL trainer\]\s+(\w+)', clean)
            if stage:
                self.stage = stage.group(1)
                if self.stage == 'rollout_start':
                    self.trajectories = ''
            streaming = re.search(r'\[RL streaming\].*?(completed=\d+/\d+.*)', clean)
            if streaming:
                self.trajectories = streaming.group(1)
            self.resize()
            self.stream.write(text+'\n')
        self.draw()

    def restore(self):
        if self.enabled:
            self.stream.write('\x1b[r')
            if self.size:
                for row in range(self.size.lines-self.footer_rows+1, self.size.lines+1):
                    self.stream.write(f'\x1b[{row};1H\x1b[2K')
                self.stream.write(f'\x1b[{self.size.lines-self.footer_rows+1};1H')
            self.stream.flush()


def main():
    log_path = Path(sys.argv[1])
    memory = ActorMemorySampler()
    display = TerminalProgress(sys.stdout, memory)
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    pending = ''

    def stop(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop)
    # The owning launcher handles Ctrl-C and closes this pipe after cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with log_path.open('ab', buffering=0) as log:
            display.draw()
            while True:
                ready, _, _ = select.select([sys.stdin.fileno()], [], [], 1)
                if not ready:
                    display.draw()
                    continue
                data = os.read(sys.stdin.fileno(), 65536)
                if not data:
                    break
                log.write(data)
                if not display.enabled:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                pending += decoder.decode(data)
                lines = re.split(r'[\r\n]', pending)
                pending = lines.pop()
                for line in lines:
                    if line:
                        if display.enabled:
                            display.line(line)
                        else:
                            display.observe(line)
                        if '[Agent0-VL trainer] actor_update_done' in ANSI.sub('', line):
                            summary = '[RL actor memory] cumulative sampled peak GiB: '+memory.text()
                            log.write((summary+'\n').encode())
                            display.line(summary)
            if display.enabled:
                pending += decoder.decode(b'', final=True)
                if pending:
                    display.line(pending)
    finally:
        memory.close()
        display.restore()
        if display.enabled:
            sys.stdout.write(display.progress+'\n'
                +f'Global tool success: {display.tool_success}\n'
                +f'Global answer accuracy: {display.answer_accuracy}\n'
                +f'Actor peak (sampled GiB): {memory.text()}\n'
                +display.step_timing+'\n')
            sys.stdout.flush()


if __name__ == '__main__':
    main()

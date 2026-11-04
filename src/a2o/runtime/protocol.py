"""持久设备进程的流式事件协议。"""
import json
import queue
import threading
import time


class Protocol:
    def __init__(self, proc, output):
        self.proc, self.output = proc, output
        self.messages = queue.Queue()
        self.events = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            with (self.output / 'protocol.log').open('w') as log:
                for line in self.proc.stdout:
                    log.write(line); log.flush()
                    if line.startswith('A2O\t'):
                        event = json.loads(line[4:])
                        self.events.append(event)
                        self.messages.put(event)
        except Exception as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(EOFError('native protocol ended'))

    def send(self, *fields):
        if any('\n' in str(f) or '\t' in str(f) for f in fields):
            raise ValueError('invalid protocol field')
        self.proc.stdin.write('\t'.join(map(str, fields)) + '\n')
        self.proc.stdin.flush()

    def wait(self, name, turn=None, timeout=180):
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('native event timeout: ' + name)
            try:
                event = self.messages.get(timeout=left)
            except queue.Empty:
                raise TimeoutError('native event timeout: ' + name) from None
            if isinstance(event, Exception):
                raise event
            if event['event'] == name and (turn is None or event['turn'] == turn):
                return event

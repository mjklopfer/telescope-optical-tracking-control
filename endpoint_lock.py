"""Cross-process exclusion for each adapter host/port.

Use platform file locking because the original adapter exposes one shared command
stream. This coordinates local processes, not separate computers on the network."""
import functools
import hashlib
import os
from pathlib import Path
import tempfile


class EndpointBusy(ConnectionError):
    pass


class EndpointLease:
    def __init__(self, host, port):
        """Choose a per-endpoint lock file in the host’s temporary directory."""
        root = Path(tempfile.gettempdir()) / ('cosmos-control-locks-' + str(getattr(os, 'getuid', lambda: 'user')()))
        root.mkdir(mode=0o700, exist_ok=True)
        key = hashlib.sha256(f'{host}:{port}'.encode()).hexdigest()
        self.path = root / (key + '.lock')
        self.file = None
        self.endpoint = f'{host}:{port}'

    def acquire(self):
        """Take a nonblocking process lock for this telescope endpoint."""
        self.file = self.path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                if not self.file.read(1):
                    self.file.write(b'0')
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            self.file = None
            raise EndpointBusy(f'{self.endpoint} is already in use by another operation or copy of this application. '
                               'Disconnect or close that connection first.') from error
        return self

    def release(self):
        """Unlock the endpoint and close its file handle, including on failure."""
        if self.file:
            try:
                if os.name == 'nt':
                    import msvcrt
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.file.close()
                self.file = None

    def __enter__(self):
        """Acquire ownership before entering the managed operation."""
        return self.acquire()

    def __exit__(self, *args):
        """Release ownership when the managed operation leaves scope."""
        self.release()


def exclusive_endpoint(operation):
    """Wrap an operation so concurrent sessions cannot share its adapter stream."""
    @functools.wraps(operation)
    def wrapped(host, port=2000, *args, **kwargs):
        """Run the decorated operation while holding the endpoint lease."""
        with EndpointLease(host, port):
            return operation(host, port, *args, **kwargs)
    return wrapped

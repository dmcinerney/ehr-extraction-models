from contextlib import contextmanager
import sys

@contextmanager
def directory(new_dir):
    sys.path.insert(1, new_dir)
    yield
    del sys.path[1]

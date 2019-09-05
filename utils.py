import os
from contextlib import contextmanager

@contextmanager
def directory(new_dir):
    original_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(original_dir)

"""Crash test file for abort/crash attribution + retry.

Has 3 tests; the middle one hard-crashes the process. After a crash, retry logic
should re-run the file with that nodeid deselected, so the 3rd test can run.
"""

import os
import signal
import time

def test_before_crash_pass():
    assert True

def test_crash_with_segfault():
    # Sleep so crash attribution doesn't get filtered by min_duration defaults.
    time.sleep(0.2)
    os.kill(os.getpid(), signal.SIGSEGV)

def test_after_crash_pass():
    # This only runs on the retry pass (with the crashing test deselected).
    assert True


# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""pytest configuration"""

import os
import pytest
import json
import threading
from datetime import datetime

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
  import jax
  import numpy
  doctest_namespace["jax"] = jax
  doctest_namespace["lax"] = jax.lax
  doctest_namespace["jnp"] = jax.numpy
  doctest_namespace["np"] = numpy


# A pytest hook that runs immediately before test collection (i.e. when pytest
# loads all the test cases to run). When running parallel tests via xdist on
# Cloud TPU, we use this hook to set the env vars needed to run multiple test
# processes across different TPU chips.
#
# It's important that the hook runs before test collection, since jax tests end
# up initializing the TPU runtime on import (e.g. to query supported test
# types). It's also important that the hook gets called by each xdist worker
# process. Luckily each worker does its own test collection.
#
# The pytest_collection hook can be used to overwrite the collection logic, but
# we only use it to set the env vars and fall back to the default collection
# logic by always returning None. See
# https://docs.pytest.org/en/latest/how-to/writing_hook_functions.html#firstresult-stop-at-first-non-none-result
# for details.
#
# The env var JAX_ENABLE_TPU_XDIST must be set for this hook to have an
# effect. We do this to minimize any effect on non-TPU tests, and as a pointer
# in test code to this "magic" hook. TPU tests should not specify more xdist
# workers than the number of TPU chips.
def pytest_collection() -> None:
  if not os.environ.get("JAX_ENABLE_TPU_XDIST", None):
    return
  # When running as an xdist worker, will be something like "gw0"
  xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
  if not xdist_worker_name.startswith("gw"):
    return
  xdist_worker_number = int(xdist_worker_name[len("gw"):])
  os.environ.setdefault("TPU_VISIBLE_CHIPS", str(xdist_worker_number))
  os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")

# Thread-safe logging for parallel test execution
class ThreadSafeTestLogger:
    def __init__(self):
        self.locks = {}
        self.global_lock = threading.Lock()
        self.base_dir = "./logs"
        os.makedirs(self.base_dir, exist_ok=True)
    
    def get_file_lock(self, test_file):
        """Get or create a lock for a specific test file"""
        with self.global_lock:
            if test_file not in self.locks:
                self.locks[test_file] = threading.Lock()
            return self.locks[test_file]
    
    def get_test_file_name(self, session):
        """Extract the test file name from the session"""
        if hasattr(session, 'config') and hasattr(session.config, 'args'):
            for arg in session.config.args:
                if arg.endswith('.py') and 'tests/' in arg:
                    return os.path.basename(arg).replace('.py', '')
        return 'unknown_test'
    
    def log_running_test(self, test_file, test_name, start_time):
        """Log the currently running test for abort detection"""
        lock = self.get_file_lock(test_file)
        with lock:
            log_data = {
                "test_file": test_file,
                "test_name": test_name,
                "start_time": start_time,
                "status": "running",
                "pid": os.getpid(),
                "gpu_id": os.environ.get("HIP_VISIBLE_DEVICES", "unknown")
            }
            
            log_file = f"{self.base_dir}/{test_file}_last_running.json"
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    def clear_running_test(self, test_file):
        """Clear the running test log when test completes successfully"""
        lock = self.get_file_lock(test_file)
        with lock:
            log_file = f"{self.base_dir}/{test_file}_last_running.json"
            if os.path.exists(log_file):
                os.remove(log_file)

# Global logger instance
test_logger = ThreadSafeTestLogger()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Hook that runs around each test to track running tests"""
    test_file = test_logger.get_test_file_name(item.session)
    test_name = item.name
    start_time = datetime.now().isoformat()
    
    # Log that this test is starting
    test_logger.log_running_test(test_file, test_name, start_time)
    
    try:
        # Run the actual test
        outcome = yield
    except Exception as e:
        # Test was interrupted/aborted - just log it, don't update reports
        print(f"Test {test_name} in {test_file} was interrupted: {str(e)}")
        raise
    else:
        # Test completed successfully (or with normal failure)
        if outcome.get_result():
            # Only clear if test completed normally (not aborted)
            test_logger.clear_running_test(test_file)

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    test_file = test_logger.get_test_file_name(session)

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished, just log any remaining aborts"""
    test_file = test_logger.get_test_file_name(session)
    
    # Check if there's a remaining running test log (indicates abort)
    log_file = f"{test_logger.base_dir}/{test_file}_last_running.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                abort_data = json.load(f)
            
            print(f"Detected aborted test: {abort_data['test_name']} in {test_file}")
            print(f"Last running file preserved for run_single_gpu.py to process: {log_file}")
            
        except Exception as e:
            print(f"Error reading abort detection file for {test_file}: {e}")

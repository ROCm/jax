#!/usr/bin/env python3
# Copyright 2025 The JAX Authors.
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

"""
Multi-GPU test runner for JAX with ROCm.
Python equivalent of [run_multi_gpu.sh]
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add the configuration directory to Python path
sys.path.insert(0, "build/rocm")

try:
    from multi_gpu_tests_config import MULTI_GPU_TESTS
    from run_single_gpu import handle_abort, generate_final_report
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

LOG_DIR = "./logs"
MAX_GPUS_PER_TEST = 8  # Limit for stability


def detect_amd_gpus():
    """Detect number of AMD/ATI GPUs using rocm-smi."""
    try:
        cmd = [
            "bash",
            "-c",
            (
                "rocm-smi | grep -E '^Device' -A 1000 | "
                "awk '$1 ~ /^[0-9]+$/ {count++} END {print count}'"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        print("Warning: Could not detect GPUs using rocm-smi, defaulting to 8")
        return 8


def cleanup_system():
    """Clean up system resources between tests."""
    print("Cleaning up system resources...")

    # Kill any remaining pytest processes
    try:
        subprocess.run(
            ["pkill", "-f", "python.*pytest"], check=False, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Wait for cleanup
    time.sleep(5)

    # Clear shared memory if possible
    try:
        subprocess.run(
            ["find", "/dev/shm", "-name", "*jax*", "-delete"],
            check=False,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Additional wait
    time.sleep(3)


def check_system_resources():
    """Check available system resources."""
    try:
        # Check memory
        result = subprocess.run(
            ["free", "-g"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "Mem:" in line:
                parts = line.split()
                available = int(parts[6])  # Available memory
                if available < 10:
                    print(f"WARNING: Low memory available: {available}GB")
                    return False
        return True
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return True  # Continue if check fails


# pylint: disable=too-many-locals
def run_multi_gpu_test(test_file, gpu_count, continue_on_fail, max_gpus=None):
    """Run a single multi-GPU test."""
    if max_gpus and gpu_count > max_gpus:
        gpu_count = max_gpus
        print(f"Limiting GPU count to {max_gpus} for stability")

    # Create GPU list (0,1,2,3 for 4 GPUs)
    gpu_list = ",".join(str(i) for i in range(gpu_count))

    # Extract test name for logging
    test_name = Path(test_file).stem

    # Setup file paths
    abs_json_log_file = os.path.abspath(f"{LOG_DIR}/multi_gpu_{test_name}_log.json")
    abs_html_log_file = os.path.abspath(f"{LOG_DIR}/multi_gpu_{test_name}_log.html")
    abs_last_running_file = os.path.abspath(f"{LOG_DIR}/{test_name}_last_running.json")

    print(f"=== Starting multi-GPU test: {test_file} ===")
    print(f"GPUs: {gpu_list} (count: {gpu_count})")
    print(f"Timestamp: {datetime.now()}")

    # Check resources before test
    if not check_system_resources():
        print("Waiting for system resources...")
        time.sleep(30)

    # Environment setup
    env = os.environ.copy()
    env.update(
        {
            "HIP_VISIBLE_DEVICES": gpu_list,
            "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
        }
    )
    # pylint: disable=duplicate-code
    # Build pytest command
    if continue_on_fail:
        cmd = [
            "python3",
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={abs_json_log_file}",
            f"--html={abs_html_log_file}",
            "--reruns",
            "3",
            "-v",
            f"./{test_file}",
        ]
    else:
        cmd = [
            "python3",
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={abs_json_log_file}",
            f"--html={abs_html_log_file}",
            "--reruns",
            "3",
            "-x",
            "-v",
            f"./{test_file}",
        ]

    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()
    try:
        # Run the test
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per test
            check=False,
        )

        duration = time.time() - start_time

        print(
            f"Test completed in {duration:.2f}s with exit code: " f"{result.returncode}"
        )

        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Handle any aborts
        success = handle_abort(
            abs_json_log_file,
            abs_html_log_file,
            abs_last_running_file,
            f"multi_gpu_{test_name}",
        )
        if success:
            print(f"Abort handling completed for {test_name}")

        return result.returncode

    except subprocess.TimeoutExpired:
        print(f"ERROR: Test {test_file} timed out after 1 hour")
        return 124  # Timeout exit code
    except (subprocess.SubprocessError, OSError) as os_e:
        print(f"ERROR: Exception running test {test_file}: {os_e}")
        return 1
    finally:
        cleanup_system()


def main():
    """Main function to run all multi-GPU tests."""
    parser = argparse.ArgumentParser(description="Run multi-GPU JAX tests")
    parser.add_argument(
        "--gpu-count", type=int, help="Number of GPUs to use (default: auto-detect)"
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=MAX_GPUS_PER_TEST,
        help=f"Maximum GPUs per test (default: {MAX_GPUS_PER_TEST})",
    )
    parser.add_argument(
        "--test-filter", type=str, help="Run only tests containing this string"
    )
    parser.add_argument(
        "-c", "--continue_on_fail", action="store_true", help="continue on failure"
    )

    args = parser.parse_args()

    # Detect GPU count if not specified
    if args.gpu_count is None:
        args.gpu_count = detect_amd_gpus()
        print(f"Detected {args.gpu_count} AMD GPUs")

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Filter tests if requested
    tests_to_run = MULTI_GPU_TESTS
    if args.test_filter:
        tests_to_run = {test for test in MULTI_GPU_TESTS if args.test_filter in test}
        print(
            f"Filtered to {len(tests_to_run)} tests containing " f"'{args.test_filter}'"
        )

    print(
        f"Running {len(tests_to_run)} multi-GPU tests with up to "
        f"{args.max_gpus} GPUs each"
    )

    # Run tests sequentially
    failed_tests = []
    passed_tests = []

    for i, test_file in enumerate(sorted(tests_to_run), 1):
        print(f"\n[{i}/{len(tests_to_run)}] Running {test_file}")

        try:
            exit_code = run_multi_gpu_test(
                test_file, args.gpu_count, args.continue_on_fail, args.max_gpus
            )

            if exit_code == 0:
                passed_tests.append(test_file)
            else:
                failed_tests.append((test_file, exit_code))
                if not args.continue_on_fail:
                    print("fail-fast: stopping after first failure")
                    break

        except KeyboardInterrupt:
            print(f"\nInterrupted during {test_file}")
            break
        except (subprocess.SubprocessError, OSError) as os_e:
            print(f"ERROR: Exception with {test_file}: {os_e}")
            failed_tests.append((test_file, -1))

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for test_file, exit_code in failed_tests:
            print(f"  {test_file} (exit code: {exit_code})")

    # Generate final report (reuse from run_single_gpu.py)
    try:
        generate_final_report()
        print("Final HTML report generated")
    except (ImportError, OSError, ValueError) as excp:
        print(f"Warning: Could not generate final report: {excp}")

    # Exit with failure if any tests failed and not continue_on_fail
    if args.continue_on_fail:
        print("continue on fail is set")
        sys.exit(0)
    sys.exit(1 if failed_tests else 0)


if __name__ == "__main__":
    # Set ROCm environment
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)

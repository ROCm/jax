#!/usr/bin/env python3
# Copyright 2022 The JAX Authors.
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

import os
import re
import csv
import argparse
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

GPU_LOCK = threading.Lock()
base_dir="./logs"

def extract_filename(path):
  base_name = os.path.basename(path)
  file_name, _ = os.path.splitext(base_name)
  return file_name

def generate_final_report_html(shell=False, env_vars={}):
  env = os.environ
  env = {**env, **env_vars}
  cmd = ["pytest_html_merger", "-i", base_dir, "-o",
          f"{base_dir}/final_compiled_report.html"]
  result = subprocess.run(cmd,
                          shell=shell,
                          capture_output=True,
                          env=env)
  if result.returncode != 0:
    cmd = " ".join(cmd)
    print("FAILED - {cmd}")
    print(result.stderr.decode())
    print(result.stdout.decode())


def generate_final_report_csv():
  merged_failed = {}
  merged_skipped = {}
  failed_report_path = os.path.join(base_dir, "final_failed_report.csv")
  skipped_report_path = os.path.join(base_dir, "final_skipped_report.csv")
  try:
    for file in os.listdir(base_dir):
      if not file.endswith(".csv"):
        continue
      with open(os.path.join(base_dir, file), mode='r',
                newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Initialize headers if the dictionaries are empty.
        if not merged_failed and not merged_skipped:
          header = reader.fieldnames
        for row in reader:
          row_id = row["id"]
          if row["status"] == "failed" and row_id not in merged_failed:
            merged_failed[row_id] = row
          elif row["status"] == "skipped" and row_id not in merged_skipped:
            merged_skipped[row_id] = row

    # Write the merged failed rows to a new CSV file.
    with open(failed_report_path, mode='w', newline='', encoding='utf-8') as file:
      writer = csv.DictWriter(file, fieldnames=header)
      writer.writeheader()
      writer.writerows(merged_failed.values())

    # Write the merged skipped rows to a new CSV file.
    with open(skipped_report_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(merged_skipped.values())

    print("Merged CSV files created successfully.")
    return 0, "Success"
  except Exception as e:
    print("Failed to generate CSV test report:", str(e))
    return 1, str(e)


def run_shell_command(cmd, shell=False, env_vars=None):
  env = os.environ.copy()
  env.update(env_vars or {})
  try:
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, env=env)
    if result.returncode != 0:
      print(f"FAILED - {' '.join(cmd)}")
      print(result.stderr)
    return result.returncode, result.stderr, result.stdout
  except Exception as e:
    print(f"Error executing command: {' '.join(cmd)}")
    print(str(e))
    return 1, str(e), ""


def collect_testmodules():
  all_test_files = []
  return_code, stderr, stdout = run_shell_command(
      ["python3", "-m", "pytest", "--collect-only", "tests"])

  if return_code != 0:
    print(stdout)
    print(stderr)
    print("Test module discovery failed.")
    exit(return_code)

  for line in stdout.split("\n"):
    match = re.match("<Module (.*)>", line.strip())
    if match:
      test_file = match.group(1)
      if "/" not in test_file:
        test_file = os.path.join("tests",test_file)
      all_test_files.append(test_file)

  print("---------- collected test modules ----------")
  print("Found %d test modules." % (len(all_test_files)))
  print("\n".join(all_test_files))
  print("--------------------------------------------")
  return all_test_files


def run_test(testmodule, gpu_tokens, continue_on_fail):
  with GPU_LOCK:
    target_gpu = gpu_tokens.pop()
  env_vars = {
      "HIP_VISIBLE_DEVICES": str(target_gpu),
      "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
  }
  testfile = extract_filename(testmodule)
  html_report_path = f"{base_dir}/{testfile}_log.html"
  csv_report_path = f"{base_dir}/{testfile}_log.csv"
  if continue_on_fail:
    cmd = ["python3", "-m", "pytest", f"--html={html_report_path}",
          "--csv", csv_report_path, "--reruns", "3", "-v", testmodule]
  else:
    cmd = ["python3", "-m", "pytest", f"--html={html_report_path}",
          "--csv", csv_report_path, "--reruns", "3", "-x", "-v", testmodule]
  _, stderr, stdout = run_shell_command(cmd, env_vars=env_vars)

  with GPU_LOCK:
    gpu_tokens.append(target_gpu)
    print("Running tests in module %s on GPU %d:" % (testmodule, target_gpu))
    print(stdout)
    print(stderr)

def run_parallel(all_testmodules, workers, continue_on_fail):
  print("Running tests with workers=", workers)
  available_gpu_tokens = list(range(workers))
  executor = ThreadPoolExecutor(max_workers=workers)
  # walking through test modules.
  for testmodule in all_testmodules:
    executor.submit(run_test, testmodule, available_gpu_tokens, continue_on_fail)
  # waiting for all jobs to finish.
  executor.shutdown(wait=True)


def find_num_gpus():
  cmd = ["lspci|grep 'controller\|accel'|grep 'AMD/ATI'|wc -l"]
  _, _, stdout = run_shell_command(cmd, shell=True)
  return int(stdout)


def main(args):
  all_testmodules = collect_testmodules()
  run_parallel(all_testmodules, args.workers, args.continue_on_fail)
  generate_final_report_html()
  generate_final_report_csv()


if __name__ == '__main__':
  os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"
  parser = argparse.ArgumentParser()
  parser.add_argument("-w",
                      "--workers",
                      type=int,
                      default=0,
                      help="Number of workers to run tests in parallel")
  parser.add_argument("-c",
                      "--continue_on_fail",
                      action="store_true",
                      help="Continue on failure")
  args = parser.parse_args()
  if args.continue_on_fail:
      print("Continue on fail is set")
  if args.workers == 0:
    sys_gpu_count = find_num_gpus()
    args.workers = sys_gpu_count
    print("%d GPUs detected." % sys_gpu_count)

  main(args)

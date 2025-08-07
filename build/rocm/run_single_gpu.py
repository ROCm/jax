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
import json
import argparse
import threading
import subprocess
import re
import html
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

GPU_LOCK = threading.Lock()
LAST_CODE = 0
base_dir = "./logs"


def extract_filename(path):
    base_name = os.path.basename(path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def combine_json_reports():
    all_json_files = [f for f in os.listdir(base_dir) if f.endswith("_log.json")]
    combined_data = []
    for json_file in all_json_files:
        with open(os.path.join(base_dir, json_file), "r") as infile:
            data = json.load(infile)
            combined_data.append(data)
    combined_json_file = f"{base_dir}/final_compiled_report.json"
    with open(combined_json_file, "w") as outfile:
        json.dump(combined_data, outfile, indent=4)

def generate_final_report(shell=False, env_vars={}):
    env = os.environ
    env = {**env, **env_vars}
    
    # First, try to merge HTML files
    cmd = [
        "pytest_html_merger",
        "-i",
        f"{base_dir}",
        "-o",
        f"{base_dir}/final_compiled_report.html",
    ]
    result = subprocess.run(cmd, shell=shell, capture_output=True, env=env)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print(result.stderr.decode())
        print("HTML merger failed, but continuing with JSON report generation...")

    # Generate json reports.
    combine_json_reports()


def run_shell_command(cmd, shell=False, env_vars={}):
    env = os.environ
    env = {**env, **env_vars}
    result = subprocess.run(cmd, shell=shell, capture_output=True, env=env)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print(result.stderr.decode())

    return result.returncode, result.stderr.decode(), result.stdout.decode()


def parse_test_log(log_file):
    """Parses the test module log file to extract test modules and functions."""
    test_files = set()
    with open(log_file, "r") as f:
        for line in f:
            report = json.loads(line)
            if "nodeid" in report:
                module = report["nodeid"].split("::")[0]
                if module and ".py" in module:
                    test_files.add(os.path.abspath(module))
    return test_files


def collect_testmodules():
    log_file = f"{base_dir}/collect_module_log.jsonl"
    return_code, stderr, stdout = run_shell_command(
        [
            "python3",
            "-m",
            "pytest",
            "--collect-only",
            "tests",
            f"--report-log={log_file}",
        ]
    )
    if return_code != 0:
        print("Test module discovery failed.")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        exit(return_code)
    print("---------- collected test modules ----------")
    test_files = parse_test_log(log_file)
    print("Found %d test modules." % (len(test_files)))
    print("--------------------------------------------")
    print("\n".join(test_files))
    return test_files


def run_test(testmodule, gpu_tokens, continue_on_fail):
    global LAST_CODE
    with GPU_LOCK:
        if LAST_CODE != 0:
            return
        target_gpu = gpu_tokens.pop()

    env_vars = {
        "HIP_VISIBLE_DEVICES": str(target_gpu),
        "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
    }
    testfile = extract_filename(testmodule)
    json_log_file = f"{base_dir}/{testfile}_log.json"
    html_log_file = f"{base_dir}/{testfile}_log.html"
    last_running_file = f"{base_dir}/{testfile}_last_running.json"

    if continue_on_fail:
        cmd = [
            "python3",
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={json_log_file}",
            f"--html={html_log_file}",
            "--reruns",
            "3",
            "-v",
            testmodule,
        ]
    else:
        cmd = [
            "python3",
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={json_log_file}",
            f"--html={html_log_file}",
            "--reruns",
            "3",
            "-x",
            "-v",
            testmodule,
        ]

    return_code, stderr, stdout = run_shell_command(cmd, env_vars=env_vars)

    # Check for aborted test log and append abort info if present
    if os.path.exists(last_running_file):
        try:
            with open(last_running_file, "r") as f:
                abort_data = json.load(f)
            start_time = datetime.fromisoformat(abort_data["start_time"])
            duration = (datetime.now() - start_time).total_seconds()
            abort_info = {
                "test_name": abort_data["test_name"],
                "reason": "Test aborted or crashed.",
                "abort_time": datetime.now().isoformat(),
                "duration": duration,
                "gpu_id": abort_data.get("gpu_id", "unknown"),
            }
            # Append to JSON log
            append_abort_to_json(json_log_file, testfile, abort_info)
            # Append to HTML log
            append_abort_to_html(html_log_file, testfile, abort_info)
            print(f"[ABORT DETECTED] {testfile}: {abort_info['test_name']}")
            # Only remove the file after successful processing
            #os.remove(last_running_file)
        except Exception as e:
            print(f"Error logging abort for {testfile}: {e}")
            # Don't remove the file if there was an error processing it

    with GPU_LOCK:
        gpu_tokens.append(target_gpu)
        if LAST_CODE == 0:
            print("Running tests in module %s on GPU %d:" % (testmodule, target_gpu))
            print(stdout)
            print(stderr)
            if continue_on_fail == False:
                LAST_CODE = return_code

def append_abort_to_json(json_file, testfile, abort_info):
    """Append abort info to JSON report in pytest format"""
    # Create test nodeid in the format expected by pytest
    test_nodeid = f"tests/{testfile}.py::{abort_info['test_name']}"
    
    abort_test = {
        "nodeid": test_nodeid,
        "lineno": 1,
        "outcome": "failed",
        "keywords": [abort_info['test_name'], testfile, "abort", ""],
        "setup": {"duration": 0.0, "outcome": "passed"},
        "call": {
            "duration": abort_info.get('duration', 0),
            "outcome": "failed",
            "longrepr": f"Test aborted: {abort_info.get('reason', 'Unknown abort reason')}\nAbort detected at: {abort_info.get('abort_time', '')}\nGPU ID: {abort_info.get('gpu_id', 'unknown')}"
        },
        "teardown": {"duration": 0.0, "outcome": "skipped"}
    }
    
    try:
        # Check if JSON file already exists (normal test run completed)
        if os.path.exists(json_file):
            # File exists - read existing data and append the aborted test
            with open(json_file, "r") as f:
                report_data = json.load(f)
            
            # Add the abort test to existing tests
            if "tests" not in report_data:
                report_data["tests"] = []
            report_data["tests"].append(abort_test)
            
            # Update summary counts
            if "summary" in report_data:
                summary = report_data["summary"]
                summary["failed"] = summary.get("failed", 0) + 1
                summary["total"] = summary.get("total", 0) + 1
                summary["collected"] = summary.get("collected", 0) + 1
                if "unskipped_total" in summary:
                    summary["unskipped_total"] = summary["unskipped_total"] + 1
            
            # Update exit code to indicate failure
            report_data["exitcode"] = 1
            
            print(f"Appended abort test to existing JSON report: {json_file}")
        else:
            # File doesn't exist - create complete pytest JSON report structure
            current_time = datetime.now().timestamp()
            report_data = {
                "created": current_time,
                "duration": abort_info.get('duration', 0),
                "exitcode": 1,  # Non-zero exit code for failure
                "root": "/rocm-jax/jax",
                "environment": {},
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "collected": 1,
                    "unskipped_total": 1
                },
                "collectors": [
                    {
                        "nodeid": "",
                        "outcome": "failed",
                        "result": [{"nodeid": f"tests/{testfile}.py", "type": "Module"}]
                    }
                ],
                "tests": [abort_test]
            }
            print(f"Created new JSON report with abort test: {json_file}")
        
        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        # Write the file
        with open(json_file, "w") as f:
            json.dump(report_data, f, indent=2)
            
    except (OSError, IOError) as e:
        print(f"Failed to write JSON report for {testfile}: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse existing JSON report for {testfile}: {e}")
        print("Creating new JSON file instead...")
        # Try creating a new file structure with just the abort test
        try:
            current_time = datetime.now().timestamp()
            new_report_data = {
                "created": current_time,
                "duration": abort_info.get('duration', 0),
                "exitcode": 1,
                "root": "/rocm-jax/jax",
                "environment": {},
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "collected": 1,
                    "unskipped_total": 1
                },
                "collectors": [
                    {
                        "nodeid": "",
                        "outcome": "failed",
                        "result": [{"nodeid": f"tests/{testfile}.py", "type": "Module"}]
                    }
                ],
                "tests": [abort_test]
            }
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            with open(json_file, "w") as f:
                json.dump(new_report_data, f, indent=2)
        except (OSError, IOError) as io_e:
            print(f"Failed to create new JSON report for {testfile}: {io_e}")

def append_abort_to_html(html_file, testfile, abort_info):
    """Generate or append abort info to pytest-html format HTML report"""
    try:
        # Check if HTML file already exists (normal test run completed)
        if os.path.exists(html_file):
            # File exists - read and append abort test row to existing HTML
            with open(html_file, "r", encoding='utf-8') as f:
                html_content = f.read()
            
            test_name = abort_info['test_name']
            duration = abort_info.get('duration', 0)
            abort_time = abort_info.get('abort_time', '')
            gpu_id = abort_info.get('gpu_id', 'unknown')
            
            # Convert duration to HH:MM:SS format matching pytest-html format
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Create abort test row HTML
            abort_row = f"""
                <tbody class="results-table-row">
                    <tr class="collapsible">
                        <td class="col-result">Failed</td>
                        <td class="col-name">tests/{testfile}.py::{test_name}</td>
                        <td class="col-duration">{duration_str}</td>
                        <td class="col-links"></td>
                    </tr>
                    <tr class="extras-row">
                        <td class="extra" colspan="4">
                            <div class="extraHTML"></div>
                            <div class="logwrapper">
                                <div class="logexpander"></div>
                                <div class="log">Test aborted: {abort_info.get('reason', 'Test aborted or crashed.')}<br/>
Abort detected at: {abort_time}<br/>
GPU ID: {gpu_id}</div>
                            </div>
                        </td>
                    </tr>
                </tbody>"""
            
            # Insert the abort row before the closing </table> tag of results-table specifically
            if '</table>' in html_content:
                # Find the results-table specifically, not the environment table
                results_table_end = html_content.find('</table>', html_content.find('<table id="results-table">'))
                if results_table_end != -1:
                    # Insert before the specific results table closing tag
                    html_content = html_content[:results_table_end] + f'{abort_row}\n    ' + html_content[results_table_end:]
                else:
                    print(f"Warning: Could not find results-table closing tag in {html_file}")
                    _create_new_html_file(html_file, testfile, abort_info)
                    return
                
                # Update the test count in the summary (find and replace pattern)
                
                # Fix malformed run-count patterns first
                malformed_pattern = r'(\d+/\d+ test done\.)'
                if re.search(malformed_pattern, html_content):
                    # Replace malformed pattern with proper format matching other pytest-html files
                    html_content = re.sub(malformed_pattern, '1 tests took 00:00:01.', html_content)
                
                # Update "X tests ran in Y" pattern that pytest_html_merger looks for (legacy format)
                count_pattern = r'(\d+) tests? ran in'
                match = re.search(count_pattern, html_content)
                if match:
                    current_count = int(match.group(1))
                    new_count = current_count + 1
                    html_content = re.sub(count_pattern, f'{new_count} tests ran in', html_content)
                
                # Update "X test took" pattern (current pytest-html format)
                count_pattern2 = r'(\d+) tests? took'
                match = re.search(count_pattern2, html_content)
                if match:
                    current_count = int(match.group(1))
                    new_count = current_count + 1
                    html_content = re.sub(count_pattern2, f'{new_count} tests took', html_content)
                
                # Update "X Failed" count in the summary
                failed_pattern = r'(\d+) Failed'
                match = re.search(failed_pattern, html_content)
                if match:
                    current_failed = int(match.group(1))
                    new_failed = current_failed + 1
                    html_content = re.sub(failed_pattern, f'{new_failed} Failed', html_content)
                else:
                    # If no failed tests before, need to enable the failed filter and update count
                    html_content = html_content.replace('0 Failed,', '1 Failed,')
                    html_content = html_content.replace('data-test-result="failed" disabled', 'data-test-result="failed"')
                
                # Update the JSON data in data-jsonblob to include the abort test
                jsonblob_pattern = r'data-jsonblob="([^"]*)"'
                match = re.search(jsonblob_pattern, html_content)
                if match:
                    try:
                        # Decode the HTML-escaped JSON
                        json_str = html.unescape(match.group(1))
                        existing_json = json.loads(json_str)
                        
                        # Add the abort test to the tests array
                        if "tests" not in existing_json:
                            existing_json["tests"] = {}
                        
                        # Create new test entry - use dictionary format for pytest_html_merger compatibility
                        test_id = f"test_{len(existing_json['tests'])}"
                        new_test = {
                            "testId": f"tests/{testfile}.py::{test_name}",
                            "id": test_id,
                            "log": f"Test aborted: {abort_info.get('reason', 'Test aborted or crashed.')}\\nAbort detected at: {abort_time}\\nGPU ID: {gpu_id}",
                            "extras": [],
                            "resultsTableRow": [
                                f'<td class="col-result">Failed</td>',
                                f'<td class="col-name">tests/{testfile}.py::{test_name}</td>',
                                f'<td class="col-duration">{duration_str}</td>',
                                f'<td class="col-links"></td>'
                            ],
                            "tableHtml": [],
                            "result": "failed",
                            "collapsed": False
                        }
                        existing_json["tests"][test_id] = new_test
                        
                        # Re-encode the JSON and escape for HTML
                        updated_json_str = html.escape(json.dumps(existing_json))
                        html_content = re.sub(jsonblob_pattern, f'data-jsonblob="{updated_json_str}"', html_content)
                        
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Warning: Could not update JSON data in HTML file: {e}")
                
                # Ensure the reload button has the hidden class to prevent "still running" message
                html_content = re.sub(r'class="summary__reload__button\s*"', 'class="summary__reload__button hidden"', html_content)
                
                with open(html_file, "w", encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"Appended abort test to existing HTML report: {html_file}")
            else:
                print(f"Warning: Could not find </table> tag in existing HTML file {html_file}")
                # Fall back to creating new file
                _create_new_html_file(html_file, testfile, abort_info)
        else:
            # File doesn't exist - create complete new HTML file
            _create_new_html_file(html_file, testfile, abort_info)
            
    except (OSError, IOError) as e:
        print(f"Failed to read/write HTML report for {testfile}: {e}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Failed to parse existing HTML report for {testfile}: {e}")
        print("Creating new HTML file instead...")
        _create_new_html_file(html_file, testfile, abort_info)

def _create_new_html_file(html_file, testfile, abort_info):
    """Create a new HTML file for abort-only report"""
    try:
        # Create the complete HTML structure matching xxx_test_log.html exactly
        test_name = abort_info['test_name']
        duration = abort_info.get('duration', 0)
        abort_time = abort_info.get('abort_time', '')
        gpu_id = abort_info.get('gpu_id', 'unknown')
        
        # Convert duration to HH:MM:SS format as expected by pytest-html
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Create JSON data for the data-container (this is what pytest_html_merger reads)
        json_data = {
            "environment": {
                "Python": "3.x",
                "Platform": "Linux", 
                "Packages": {"pytest": "8.4.1", "pluggy": "1.6.0"},
                "Plugins": {"rerunfailures": "15.1", "json-report": "1.5.0", "html": "4.1.1", "reportlog": "0.4.0", "metadata": "3.1.1", "hypothesis": "6.136.6"}
            },
            "tests": {
                "test_0": {
                    "testId": f"tests/{testfile}.py::{test_name}",
                    "id": "test_0",
                    "log": f"Test aborted: {abort_info.get('reason', 'Test aborted or crashed.')}\\nAbort detected at: {abort_time}\\nGPU ID: {gpu_id}",
                    "extras": [],
                    "resultsTableRow": [
                        f'<td class="col-result">Failed</td>',
                        f'<td class="col-name">tests/{testfile}.py::{test_name}</td>',
                        f'<td class="col-duration">{duration_str}</td>',
                        f'<td class="col-links"></td>'
                    ],
                    "tableHtml": [],
                    "result": "failed",
                    "collapsed": False
                }
            },
            "renderCollapsed": ["passed"],
            "initialSort": "result",
            "title": f"{testfile}_log.html"
        }
        
        # Convert JSON to HTML-escaped string for data-jsonblob attribute
        json_blob = html.escape(json.dumps(json_data))
        
        html_content = f'''<!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8"/>
            <title id="head-title">{testfile}_log.html</title>
            <link href="assets/style.css" rel="stylesheet" type="text/css"/>
          </head>
          <body onLoad="init()">
            <h1 id="title">{testfile}_log.html</h1>
            <p>Report generated on {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')} by <a href="https://pypi.python.org/pypi/pytest-html">pytest-html</a> v4.1.1</p>
            <div id="environment-header">
              <h2>Environment</h2>
            </div>
            <table id="environment"></table>
            <div class="summary">
              <div class="summary__data">
                <h2>Summary</h2>
                <div class="additional-summary prefix">
                </div>
                <p class="run-count">1 tests took {duration_str}.</p>
                <p class="filter">(Un)check the boxes to filter the results.</p>
                <div class="summary__reload">
                  <div class="summary__reload__button hidden" onclick="location.reload()">
                    <div>There are still tests running. <br />Reload this page to get the latest results!</div>
                  </div>
                </div>
                <div class="summary__spacer"></div>
                <div class="controls">
                  <div class="filters">
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="failed" />
                    <span class="failed">1 Failed,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="passed" disabled/>
                    <span class="passed">0 Passed,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="skipped" disabled/>
                    <span class="skipped">0 Skipped,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="xfailed" disabled/>
                    <span class="xfailed">0 Expected failures,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="xpassed" disabled/>
                    <span class="xpassed">0 Unexpected passes,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="error" disabled/>
                    <span class="error">0 Errors,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="rerun" disabled/>
                    <span class="rerun">0 Reruns</span>
                  </div>
                  <div class="collapse">
                    <button id="show_all_details">Show all details</button>&nbsp;/&nbsp;<button id="hide_all_details">Hide all details</button>
                  </div>
                </div>
              </div>
              <div class="additional-summary summary">
              </div>
              <div class="additional-summary postfix">
              </div>
            </div>
            <table id="results-table">
              <thead id="results-table-head">
                <tr>
                  <th class="sortable result initial-sort" data-column-type="result">Result</th>
                  <th class="sortable" data-column-type="name">Test</th>
                  <th class="sortable" data-column-type="duration">Duration</th>
                  <th class="sortable links" data-column-type="links">Links</th>
                </tr>
              </thead>
              <tbody class="results-table-row">
                <tr class="collapsible">
                  <td class="col-result">Failed</td>
                  <td class="col-name">tests/{testfile}.py::{test_name}</td>
                  <td class="col-duration">{duration_str}</td>
                  <td class="col-links"></td>
                </tr>
                <tr class="extras-row">
                  <td class="extra" colspan="4">
                    <div class="extraHTML"></div>
                    <div class="logwrapper">
                      <div class="logexpander"></div>
                      <div class="log">Test aborted: {abort_info.get('reason', 'Test aborted or crashed.')}<br/>
        Abort detected at: {abort_time}<br/>
        GPU ID: {gpu_id}</div>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
            <div id="data-container" data-jsonblob="{json_blob}"></div>
            <script>
              function init() {{
                // Initialize any required functionality
              }}
            </script>
          </body>
        </html>'''
        
        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(html_file), exist_ok=True)
        
        # Write the HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Created new HTML report: {html_file}")
        
    except (OSError, IOError) as e:
        print(f"Failed to write new HTML report for {testfile}: {e}")
    except Exception as e:
        print(f"Unexpected error creating new HTML report for {testfile}: {e}")
        traceback.print_exc()



def run_parallel(all_testmodules, p, c):
    print(f"Running tests with parallelism = {p}")
    available_gpu_tokens = list(range(p))
    executor = ThreadPoolExecutor(max_workers=p)
    # walking through test modules.
    for testmodule in all_testmodules:
        executor.submit(run_test, testmodule, available_gpu_tokens, c)
    # waiting for all modules to finish.
    executor.shutdown(wait=True)


def find_num_gpus():
    cmd = [r"lspci|grep 'controller\|accel'|grep 'AMD/ATI'|wc -l"]
    _, _, stdout = run_shell_command(cmd, shell=True)
    return int(stdout)


def main(args):
    all_testmodules = collect_testmodules()
    run_parallel(all_testmodules, args.parallel, args.continue_on_fail)
    generate_final_report()
    exit(LAST_CODE)


if __name__ == "__main__":
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--parallel", type=int, help="number of tests to run in parallel"
    )
    parser.add_argument(
        "-c", "--continue_on_fail", action="store_true", help="continue on failure"
    )
    args = parser.parse_args()
    if args.continue_on_fail:
        print("continue on fail is set")
    if args.parallel is None:
        sys_gpu_count = find_num_gpus()
        args.parallel = sys_gpu_count
        print("%d GPUs detected." % sys_gpu_count)

    main(args)



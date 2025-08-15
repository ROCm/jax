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

"""
Single GPU test runner for JAX with ROCm.

This script runs JAX tests on individual GPUs in parallel, excluding
multi-GPU tests that require multiple devices.
"""

import os
import sys
import json
import argparse
import threading
import subprocess
import re
import html
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Add the configuration directory to Python path
sys.path.insert(0, "build/rocm")

# Import multi-GPU tests configuration
try:
    from multi_gpu_tests_config import MULTI_GPU_TESTS
except ImportError as e:
    print(
        f"Error: Could not import multi_gpu_tests_config from build/rocm/: {e}"
    )
    print(
        "Please ensure multi_gpu_tests_config.py exists in build/rocm/ directory"
    )
    sys.exit(1)

GPU_LOCK = threading.Lock()
LAST_CODE = 0
BASE_DIR = "./logs"


def extract_filename(path):
    """Extract filename without extension from a path."""
    base_name = os.path.basename(path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def combine_json_reports():
    """Combine all individual JSON test reports into a single report."""
    all_json_files = [f for f in os.listdir(BASE_DIR) if f.endswith("_log.json")]
    combined_data = []
    for json_file in all_json_files:
        with open(os.path.join(BASE_DIR, json_file), "r", encoding="utf-8") as infile:
            data = json.load(infile)
            combined_data.append(data)
    combined_json_file = f"{BASE_DIR}/final_compiled_report.json"
    with open(combined_json_file, "w", encoding="utf-8") as outfile:
        json.dump(combined_data, outfile, indent=4)


def generate_final_report(shell=False, env_vars=None):
    """Generate final HTML and JSON reports by merging individual test reports."""
    if env_vars is None:
        env_vars = {}
    env = os.environ.copy()
    env.update(env_vars)

    # First, try to merge HTML files
    cmd = [
        "pytest_html_merger",
        "-i",
        f"{BASE_DIR}",
        "-o",
        f"{BASE_DIR}/final_compiled_report.html",
    ]
    result = subprocess.run(cmd, shell=shell, capture_output=True, env=env, check=False)
    if result.returncode != 0:
        print(f"FAILED - {' '.join(cmd)}")
        print(result.stderr.decode())
        print("HTML merger failed, but continuing with JSON report generation...")

    # Generate json reports.
    combine_json_reports()


def run_shell_command(cmd, shell=False, env_vars=None):
    """Run a shell command and return the result."""
    if env_vars is None:
        env_vars = {}
    env = os.environ.copy()
    env.update(env_vars)
    result = subprocess.run(cmd, shell=shell, capture_output=True, env=env, check=False)
    if result.returncode != 0:
        print(f"FAILED - {' '.join(cmd)}")
        print(result.stderr.decode())

    return result.returncode, result.stderr.decode(), result.stdout.decode()


def parse_test_log(log_file):
    """Parse the test module log file to extract test modules and functions."""
    test_files = set()
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            report = json.loads(line)
            if "nodeid" in report:
                module = report["nodeid"].split("::")[0]
                if module and ".py" in module:
                    test_files.add(os.path.abspath(module))
    return test_files


def collect_testmodules():
    """Collect all test modules, excluding multi-GPU tests."""
    log_file = f"{BASE_DIR}/collect_module_log.jsonl"
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
        sys.exit(return_code)
    print("---------- collected test modules ----------")
    test_files = parse_test_log(log_file)

    # Filter out multi-GPU tests
    filtered_test_files = set()
    excluded_count = 0
    for test_file in test_files:
        # Convert absolute path to relative path for comparison
        relative_path = os.path.relpath(test_file)
        if relative_path not in MULTI_GPU_TESTS:
            filtered_test_files.add(test_file)
        else:
            excluded_count += 1
            print(f"Excluding multi-GPU test: {relative_path}")

    print(f"Found {len(filtered_test_files)} test modules.")
    print(f"Excluded {excluded_count} multi-GPU test modules.")
    print("--------------------------------------------")
    print("\n".join(filtered_test_files))
    return filtered_test_files


def run_test(testmodule, gpu_tokens, continue_on_fail):
    """Run a single test module on an available GPU."""
    global LAST_CODE  # pylint: disable=global-statement
    with GPU_LOCK:
        if LAST_CODE != 0:
            return
        target_gpu = gpu_tokens.pop()

    env_vars = {
        "HIP_VISIBLE_DEVICES": str(target_gpu),
        "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
    }
    testfile = extract_filename(testmodule)
    json_log_file = f"{BASE_DIR}/{testfile}_log.json"
    html_log_file = f"{BASE_DIR}/{testfile}_log.html"
    last_running_file = f"{BASE_DIR}/{testfile}_last_running.json"

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
    if handle_abort(json_log_file, html_log_file, last_running_file, testfile):
        # Abort was detected and handled
        pass

    with GPU_LOCK:
        gpu_tokens.append(target_gpu)
        if LAST_CODE == 0:
            print(f"Running tests in module {testmodule} on GPU {target_gpu}:")
            print(stdout)
            print(stderr)
            if not continue_on_fail:
                LAST_CODE = return_code


def append_abort_to_json(json_file, testfile, abort_info):
    # pylint: disable=too-many-locals
    """Append abort info to JSON report in pytest format."""
    # Use the test_name which could be a full nodeid or just a test name
    test_identifier = abort_info["test_name"]
    test_class = abort_info.get("test_class", "UnknownClass")

    # If test_identifier is already a full nodeid, use it; otherwise construct one
    if "::" in test_identifier and test_identifier.startswith("tests/"):
        # Already a full nodeid like "tests/image_test.py::ImageTest::testMethod"
        test_nodeid = test_identifier
    elif "::" in test_identifier:
        # Partial nodeid like "ImageTest::testMethod" - add the file part
        test_nodeid = f"tests/{testfile}.py::{test_identifier}"
    else:
        # Just a test method name
        test_nodeid = f"tests/{testfile}.py::{test_identifier}"

    abort_longrepr = (
        f"Test aborted: {abort_info.get('reason', 'Unknown abort reason')}\n"
        f"Test Class: {test_class}\n"
        f"Abort detected at: {abort_info.get('abort_time', '')}\n"
        f"GPU ID: {abort_info.get('gpu_id', 'unknown')}"
    )

    abort_test = {
        "nodeid": test_nodeid,
        "lineno": 1,
        "outcome": "failed",
        "keywords": [abort_info["test_name"], testfile, "abort", test_class, ""],
        "setup": {"duration": 0.0, "outcome": "passed"},
        "call": {
            "duration": abort_info.get("duration", 0),
            "outcome": "failed",
            "longrepr": abort_longrepr,
        },
        "teardown": {"duration": 0.0, "outcome": "skipped"},
    }

    try:
        # Check if JSON file already exists (normal test run completed)
        if os.path.exists(json_file):
            # File exists - read existing data and append the aborted test
            with open(json_file, "r", encoding="utf-8") as f:
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
                "duration": abort_info.get("duration", 0),
                "exitcode": 1,  # Non-zero exit code for failure
                "root": "/rocm-jax/jax",
                "environment": {},
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "collected": 1,
                    "unskipped_total": 1,
                },
                "collectors": [
                    {
                        "nodeid": "",
                        "outcome": "failed",
                        "result": [
                            {"nodeid": f"tests/{testfile}.py", "type": "Module"}
                        ],
                    }
                ],
                "tests": [abort_test],
            }
            print(f"Created new JSON report with abort test: {json_file}")

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        # Write the file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    except (OSError, IOError) as io_err:
        print(f"Failed to write JSON report for {testfile}: {io_err}")
    except json.JSONDecodeError as json_err:
        print(f"Failed to parse existing JSON report for {testfile}: {json_err}")
        print("Creating new JSON file instead...")
        # Try creating a new file structure with just the abort test
        try:
            current_time = datetime.now().timestamp()
            new_report_data = {
                "created": current_time,
                "duration": abort_info.get("duration", 0),
                "exitcode": 1,
                "root": "/rocm-jax/jax",
                "environment": {},
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "collected": 1,
                    "unskipped_total": 1,
                },
                "collectors": [
                    {
                        "nodeid": "",
                        "outcome": "failed",
                        "result": [
                            {"nodeid": f"tests/{testfile}.py", "type": "Module"}
                        ],
                    }
                ],
                "tests": [abort_test],
            }
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(new_report_data, f, indent=2)
        except (OSError, IOError) as io_e:
            print(f"Failed to create new JSON report for {testfile}: {io_e}")


def _create_abort_row_html(testfile, abort_info):
    """Create HTML row for abort test."""
    test_identifier = abort_info["test_name"]
    test_class = abort_info.get("test_class", "UnknownClass")

    # If test_identifier is already a full nodeid, use it; otherwise construct one
    if "::" in test_identifier and test_identifier.startswith("tests/"):
        # Already a full nodeid like "tests/image_test.py::ImageTest::testMethod"
        display_name = test_identifier
    elif "::" in test_identifier:
        # Partial nodeid like "ImageTest::testMethod" - add the file part
        display_name = f"tests/{testfile}.py::{test_identifier}"
    else:
        # Just a test method name
        display_name = f"tests/{testfile}.py::{test_identifier}"

    duration = abort_info.get("duration", 0)
    abort_time = abort_info.get("abort_time", "")
    gpu_id = abort_info.get("gpu_id", "unknown")

    # Convert duration to HH:MM:SS format matching pytest-html format
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    abort_reason = abort_info.get("reason", "Test aborted or crashed.")
    log_content = (
        f"Test aborted: {abort_reason}<br/>"
        f"Test Class: {test_class}<br/>"
        f"Abort detected at: {abort_time}<br/>"
        f"GPU ID: {gpu_id}"
    )

    return f"""
                <tbody class="results-table-row">
                    <tr class="collapsible">
                        <td class="col-result">Failed</td>
                        <td class="col-name">{display_name}</td>
                        <td class="col-duration">{duration_str}</td>
                        <td class="col-links"></td>
                    </tr>
                    <tr class="extras-row">
                        <td class="extra" colspan="4">
                            <div class="extraHTML"></div>
                            <div class="logwrapper">
                                <div class="logexpander"></div>
                                <div class="log">{log_content}</div>
                            </div>
                        </td>
                    </tr>
                </tbody>"""


def _update_html_summary_counts(html_content):
    """Update test counts in HTML summary."""
    # Fix malformed run-count patterns first
    malformed_pattern = r"(\d+/\d+ test done\.)"
    if re.search(malformed_pattern, html_content):
        # Replace malformed pattern with proper format
        html_content = re.sub(malformed_pattern, "1 tests took 00:00:01.", html_content)

    # Update "X tests ran in Y" pattern (legacy format)
    count_pattern = r"(\d+) tests? ran in"
    match = re.search(count_pattern, html_content)
    if match:
        current_count = int(match.group(1))
        new_count = current_count + 1
        html_content = re.sub(count_pattern, f"{new_count} tests ran in", html_content)

    # Update "X test took" pattern (current pytest-html format)
    count_pattern2 = r"(\d+) tests? took"
    match = re.search(count_pattern2, html_content)
    if match:
        current_count = int(match.group(1))
        new_count = current_count + 1
        html_content = re.sub(count_pattern2, f"{new_count} tests took", html_content)

    # Update "X Failed" count in the summary
    failed_pattern = r"(\d+) Failed"
    match = re.search(failed_pattern, html_content)
    if match:
        current_failed = int(match.group(1))
        new_failed = current_failed + 1
        html_content = re.sub(failed_pattern, f"{new_failed} Failed", html_content)
    else:
        # If no failed tests before, enable the failed filter and update count
        html_content = html_content.replace("0 Failed,", "1 Failed,")
        html_content = html_content.replace(
            'data-test-result="failed" disabled',
            'data-test-result="failed"',
        )

    return html_content


def _update_html_json_data(
    html_content, testfile, abort_info
):  # pylint: disable=too-many-locals
    """Update JSON data in HTML file."""
    jsonblob_pattern = r'data-jsonblob="([^"]*)"'
    match = re.search(jsonblob_pattern, html_content)
    if not match:
        return html_content

    try:
        # Decode the HTML-escaped JSON
        json_str = html.unescape(match.group(1))
        existing_json = json.loads(json_str)

        # Add the abort test to the tests array
        if "tests" not in existing_json:
            existing_json["tests"] = {}

        test_name = abort_info["test_name"]
        duration = abort_info.get("duration", 0)
        abort_time = abort_info.get("abort_time", "")
        gpu_id = abort_info.get("gpu_id", "unknown")

        # Convert duration to HH:MM:SS format
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create new test entry
        test_id = f"test_{len(existing_json['tests'])}"
        log_msg = (
            f"Test aborted: {abort_info.get('reason', 'Test aborted or crashed.')}\\n"
            f"Abort detected at: {abort_time}\\n"
            f"GPU ID: {gpu_id}"
        )
        new_test = {
            "testId": test_name if ("::" in test_name and test_name.startswith("tests/")) else f"tests/{testfile}.py::{test_name}",
            "id": test_id,
            "log": log_msg,
            "extras": [],
            "resultsTableRow": [
                '<td class="col-result">Failed</td>',
                f'<td class="col-name">{test_name if ("::" in test_name and test_name.startswith("tests/")) else f"tests/{testfile}.py::{test_name}"}</td>',
                f'<td class="col-duration">{duration_str}</td>',
                '<td class="col-links"></td>',
            ],
            "tableHtml": [],
            "result": "failed",
            "collapsed": False,
        }
        existing_json["tests"][test_id] = new_test

        # Re-encode the JSON and escape for HTML
        updated_json_str = html.escape(json.dumps(existing_json))
        html_content = re.sub(
            jsonblob_pattern,
            f'data-jsonblob="{updated_json_str}"',
            html_content,
        )

    except ( # pylint: disable=broad-exception-caught
        json.JSONDecodeError,
        Exception,
    ) as ex:  # pylint: disable=broad-exception-caught
        print(f"Warning: Could not update JSON data in HTML file: {ex}")

    return html_content


def append_abort_to_html(html_file, testfile, abort_info):
    """Generate or append abort info to pytest-html format HTML report."""
    try:
        # Check if HTML file already exists (normal test run completed)
        if os.path.exists(html_file):
            # File exists - read and append abort test row to existing HTML
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Create abort test row HTML
            abort_row = _create_abort_row_html(testfile, abort_info)

            # Insert the abort row before the closing </table> tag
            if "</table>" in html_content:
                # Find the results-table specifically, not the environment table
                results_table_start = html_content.find('<table id="results-table">')
                results_table_end = html_content.find("</table>", results_table_start)
                if results_table_end != -1:
                    # Insert before the specific results table closing tag
                    html_content = (
                        html_content[:results_table_end]
                        + f"{abort_row}\n    "
                        + html_content[results_table_end:]
                    )
                else:
                    print(
                        f"Warning: Could not find results-table closing tag in {html_file}"
                    )
                    _create_new_html_file(html_file, testfile, abort_info)
                    return

                # Update test counts and JSON data
                html_content = _update_html_summary_counts(html_content)
                html_content = _update_html_json_data(
                    html_content, testfile, abort_info
                )

                # Ensure the reload button has the hidden class
                html_content = re.sub(
                    r'class="summary__reload__button\s*"',
                    'class="summary__reload__button hidden"',
                    html_content,
                )

                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html_content)

                print(f"Appended abort test to existing HTML report: {html_file}")
            else:
                print(
                    f"Warning: Could not find </table> tag in existing HTML file {html_file}"
                )
                # Fall back to creating new file
                _create_new_html_file(html_file, testfile, abort_info)
        else:
            # File doesn't exist - create complete new HTML file
            _create_new_html_file(html_file, testfile, abort_info)

    except (OSError, IOError) as io_err:
        print(f"Failed to read/write HTML report for {testfile}: {io_err}")
    except (json.JSONDecodeError, UnicodeDecodeError) as parse_err:
        print(f"Failed to parse existing HTML report for {testfile}: {parse_err}")
        print("Creating new HTML file instead...")
        _create_new_html_file(html_file, testfile, abort_info)


def _create_new_html_file(
    html_file, testfile, abort_info
):  # pylint: disable=too-many-locals
    """Create a new HTML file for abort-only report."""
    try:
        # Create the complete HTML structure matching pytest-html format
        test_name = abort_info["test_name"]
        duration = abort_info.get("duration", 0)
        abort_time = abort_info.get("abort_time", "")
        gpu_id = abort_info.get("gpu_id", "unknown")

        # Convert duration to HH:MM:SS format as expected by pytest-html
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create JSON data for the data-container
        abort_reason = abort_info.get("reason", "Test aborted or crashed.")
        log_msg = (
            f"Test aborted: {abort_reason}\\n"
            f"Abort detected at: {abort_time}\\n"
            f"GPU ID: {gpu_id}"
        )
        json_data = {
            "environment": {
                "Python": "3.x",
                "Platform": "Linux",
                "Packages": {"pytest": "8.4.1", "pluggy": "1.6.0"},
                "Plugins": {
                    "rerunfailures": "15.1",
                    "json-report": "1.5.0",
                    "html": "4.1.1",
                    "reportlog": "0.4.0",
                    "metadata": "3.1.1",
                    "hypothesis": "6.136.6",
                },
            },
            "tests": {
                "test_0": {
                    "testId": test_name if ("::" in test_name and test_name.startswith("tests/")) else f"tests/{testfile}.py::{test_name}",
                    "id": "test_0",
                    "log": log_msg,
                    "extras": [],
                    "resultsTableRow": [
                        '<td class="col-result">Failed</td>',
                        f'<td class="col-name">{test_name if ("::" in test_name and test_name.startswith("tests/")) else f"tests/{testfile}.py::{test_name}"}</td>',
                        f'<td class="col-duration">{duration_str}</td>',
                        '<td class="col-links"></td>',
                    ],
                    "tableHtml": [],
                    "result": "failed",
                    "collapsed": False,
                }
            },
            "renderCollapsed": ["passed"],
            "initialSort": "result",
            "title": f"{testfile}_log.html",
        }

        # Convert JSON to HTML-escaped string for data-jsonblob attribute
        json_blob = html.escape(json.dumps(json_data))

        # Create the HTML content
        current_time_str = datetime.now().strftime("%d-%b-%Y at %H:%M:%S")
        log_content = (
            f"Test aborted: {abort_reason}<br/>"
            f"Abort detected at: {abort_time}<br/>"
            f"GPU ID: {gpu_id}"
        )

        html_content = _generate_html_template(
            {
                "testfile": testfile,
                "test_name": test_name,
                "duration_str": duration_str,
                "current_time_str": current_time_str,
                "log_content": log_content,
                "json_blob": json_blob,
            }
        )

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(html_file), exist_ok=True)

        # Write the HTML file
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Created new HTML report: {html_file}")

    except (OSError, IOError) as io_err:
        print(f"Failed to write new HTML report for {testfile}: {io_err}")
    except Exception as ex:  # pylint: disable=broad-exception-caught
        print(f"Unexpected error creating new HTML report for {testfile}: {ex}")
        traceback.print_exc()


def _generate_html_template(template_data):
    """Generate HTML template for test report."""
    testfile = template_data["testfile"]
    test_name = template_data["test_name"]
    duration_str = template_data["duration_str"]
    current_time_str = template_data["current_time_str"]
    log_content = template_data["log_content"]
    json_blob = template_data["json_blob"]
    return f"""<!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8"/>
            <title id="head-title">{testfile}_log.html</title>
            <link href="assets/style.css" rel="stylesheet" type="text/css"/>
          </head>
          <body onLoad="init()">
            <h1 id="title">{testfile}_log.html</h1>
            <p>Report generated on {current_time_str} by
               <a href="https://pypi.python.org/pypi/pytest-html">pytest-html</a> v4.1.1</p>
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
                  <td class="col-name">{test_name}</td>
                  <td class="col-duration">{duration_str}</td>
                  <td class="col-links"></td>
                </tr>
                <tr class="extras-row">
                  <td class="extra" colspan="4">
                    <div class="extraHTML"></div>
                    <div class="logwrapper">
                      <div class="logexpander"></div>
                      <div class="log">{log_content}</div>
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
        </html>"""


def handle_abort(json_file, html_file, last_running_file, testfile):
    """Handle abort detection and append abort info to both JSON and HTML reports."""
    if not os.path.exists(last_running_file):
        return False

    try:
        with open(last_running_file, "r", encoding="utf-8") as f:
            abort_data = json.load(f)
        start_time = datetime.fromisoformat(abort_data["start_time"])
        duration = (datetime.now() - start_time).total_seconds()

        # Use nodeid if available (includes class and method names),
        # otherwise fall back to test_name
        test_identifier = abort_data.get(
            "nodeid", abort_data.get("test_name", "unknown_test")
        )

        # Extract test class name
        test_class = "UnknownClass"
        if "::" in test_identifier:
            parts = test_identifier.split("::")
            if len(parts) >= 3:
                test_class = parts[1]  # TestClass
            elif len(parts) == 2:
                test_class = parts[1]  # Could be just test_method

        abort_info = {
            "test_name": test_identifier,
            "test_class": test_class,
            "reason": "Test aborted or crashed.",
            "abort_time": datetime.now().isoformat(),
            "duration": duration,
            "gpu_id": abort_data.get("gpu_id", "unknown"),
        }
        # Append to JSON log
        append_abort_to_json(json_file, testfile, abort_info)
        # Append to HTML log
        append_abort_to_html(html_file, testfile, abort_info)
        print(f"[ABORT DETECTED] {testfile}: {abort_info['test_name']}")
        print(f"  Test Class: {test_class}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  GPU ID: {abort_info['gpu_id']}")
        # Remove the last running file after successful processing
        # os.remove(last_running_file)
        return True
    except Exception as ex:  # pylint: disable=broad-exception-caught
        print(f"Error logging abort for {testfile}: {ex}")
        # Don't remove the file if there was an error processing it
        return False


def run_parallel(all_testmodules, p, c):
    """Run tests in parallel across multiple GPUs."""
    print(f"Running tests with parallelism = {p}")
    available_gpu_tokens = list(range(p))
    executor = ThreadPoolExecutor(max_workers=p)
    # walking through test modules.
    for testmodule in all_testmodules:
        executor.submit(run_test, testmodule, available_gpu_tokens, c)
    # waiting for all modules to finish.
    executor.shutdown(wait=True)


def find_num_gpus():
    """Find the number of AMD/ATI GPUs available."""
    cmd = [r"rocm-smi | grep -E '^Device' -A 1000 | awk '$1 ~ /^[0-9]+$/ {count++} END {print count}'"]
    _, _, stdout = run_shell_command(cmd, shell=True)
    return int(stdout)


def main(args):
    """Main function to run all test modules."""
    all_testmodules = collect_testmodules()
    run_parallel(all_testmodules, args.parallel, args.continue_on_fail)
    generate_final_report()
    sys.exit(LAST_CODE)


if __name__ == "__main__":
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--parallel", type=int, help="number of tests to run in parallel"
    )
    parser.add_argument(
        "-c", "--continue_on_fail", action="store_true", help="continue on failure"
    )
    parsed_args = parser.parse_args()
    if parsed_args.continue_on_fail:
        print("continue on fail is set")
    if parsed_args.parallel is None:
        sys_gpu_count = find_num_gpus()
        parsed_args.parallel = sys_gpu_count
        print(f"{sys_gpu_count} GPUs detected.")

    main(parsed_args)


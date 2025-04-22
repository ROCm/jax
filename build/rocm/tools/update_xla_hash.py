"""Update the third_party/xla/workspace.bzl file to use the given XLA commit"""

import argparse
import logging
import os.path
import re
import subprocess

logger = logging.getLogger(__name__)


def update_xla_hash(xla_commit, xla_repo, workspace_file_path):
    # Verify that the workspace_file exists
    if not os.path.isfile(workspace_file_path):
        raise ValueError(f"Workspace file '{workspace_file}' does not exist")

    # Convert xla_commit to a commit hash if it's a branch
    xla_commit_hash = xla_commit

    # Get the sha256 of this commit
    curl_proc = subprocess.Popen(
        args=[
            "curl",
            "--output",
            "-",
            "-L",
            f"{xla_repo}/archive/{xla_commit_hash}.tar.gz",
        ],
        stdout=subprocess.PIPE,
    )
    sha256_proc = subprocess.Popen(
        args=["sha256sum"],
        stdin=curl_proc.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )
    sha256_text, _ = sha256_proc.communicate()
    # sha256sum sticks some extra characters onto its output. Just take the
    # first 64, since sha256s area always 64 characters long.
    sha256_text = sha256_text[:64]
    print(sha256_text)

    # Open the workspace file
    with open(workspace_file_path, "r+") as workspace_file:
        contents = workspace_file.read()
        # Edit the commit hash, sha256 hash, and repo
        contents = re.sub(
            'XLA_COMMIT = "[a-z0-9]*"',
            f'XLA_COMIT = "{xla_commit_hash}"',
            contents,
            flags=re.M,
        )
        contents = re.sub(
            'XLA_SHA256 = "[a-z0-9]*"',
            f'XLA_SHA256 = "{sha256_text}"',
            contents,
            flags=re.M,
        )
        contents = re.sub(
            'tf_mirror_urls\("[a-zA-Z0-9:/.]+/archive',
            f'tf_mirror_urls\("{xla_repo}/archive',
            contents,
            flags=re.M,
        )
        # Write to the workspace file
        workspace_file.seek(0)
        workspace_file.write(contents)
        workspace_file.truncate()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Update the XLA commit hash in the workspace.bzl file"
    )
    arg_parser.add_argument(
        "xla_commit",
        help="Branch or commit to put in the workspace file",
    )
    arg_parser.add_argument(
        "--xla-repo",
        default="https://github.com/openxla/xla",
        help="URL to the XLA repo where this branch or commit can be "
        "found. Defaults to https://github.com/openxla/xla.",
    )
    arg_parser.add_argument(
        "--workspace-file",
        default="./third_party/xla/workspace.bzl",
        help="Path to the workspace.bzl file to put the hash. Defaults to "
        "./third_party/xla/workspace.bzl.",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    update_xla_hash(args.xla_commit, args.xla_repo, args.workspace_file)

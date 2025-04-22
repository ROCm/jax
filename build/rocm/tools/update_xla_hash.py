"""Update the third_party/xla/workspace.bzl file to use the given XLA commit"""

import argparse
import logging
import os.path
import re
import requests
import subprocess

GH_COMMIT_URL = "https://api.github.com/repos/{0}/commits/{1}"
GH_BASE_URL = "https://github.com"
logger = logging.getLogger(__name__)


def update_xla_hash(xla_commit, xla_repo, workspace_file_path, gh_token):
    # Verify that the workspace_file exists
    if not os.path.isfile(workspace_file_path):
        raise ValueError(f"Workspace file '{workspace_file}' does not exist")

    # If we were given a GH auth token, use it to make sure that the commit
    # exists and convert a branch name to a commit hash
    if gh_token:
        logger.debug(GH_COMMIT_URL.format(xla_repo, xla_commit))
        commit_info_resp = requests.get(
            url=GH_COMMIT_URL.format(xla_repo, xla_commit),
            headers={
                "Accept": "application/vnd.github.sha",
                "Authorization": f"Bearer {gh_token}",
                "X-Github-Api-Version": "2022-11-28",
            }
        )
        commit_info_resp.raise_for_status()
        logger.info("Found commit hash via GH API: %s", commit_info_resp.text)
        xla_commit_hash = commit_info_resp.text
    # If the user didn't give us a token make sure the commit hash looks hashy
    else:
        if not xla_commit.isalnum():
            raise ValueError(f"XLA commit hash '{xla_commit}' is not a valid commit hash")
        xla_commit_hash = xla_commit

    # Get the sha256 of this commit
    curl_proc = subprocess.Popen(
        args=[
            "curl",
            "--output",
            "-",
            "-L",
            f"{GH_BASE_URL}/{xla_repo}/archive/{xla_commit_hash}.tar.gz",
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
            f'tf_mirror_urls("{GH_BASE_URL}/{xla_repo}/archive',
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
        "gh_token",
        help="Github token to authenticate with. Either the GIHUB_TOKEN from Actions or your PAT."
    )
    arg_parser.add_argument(
        "--xla-repo",
        default="openxla/xla",
        help="The repo where this branch or commit can be found. Should be in the form of <owner>/<repo>. Defaults to openxla/xla.",
    )
    arg_parser.add_argument(
        "--workspace-file",
        default="./third_party/xla/workspace.bzl",
        help="Path to the workspace.bzl file to put the hash. Defaults to ./third_party/xla/workspace.bzl.",
    )
    arg_parser.add_argument(
        "-v", "--verbose",
        help="Turn on debug logging",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=args.loglevel)
    update_xla_hash(args.xla_commit, args.xla_repo, args.workspace_file, args.gh_token)

# ROCm CI Dev Guide

This guide lays out how to do some dev operations, what branches live in this repo, and what CI
workflows live in this repo.

# Processes

## Making a Change

1. Clone `rocm/jax` and check out the `rocm-main` branch.
2. Create a new feature branch with `git checkout -b <my feature name>`.
3. Make your changes on the feature branch and test them locally.
4. Push your changes to a new feature branch in `rocm/jax` by running
   `git push orgin HEAD`.
5. Open a PR from your new feature branch into `rocm-main` with a nice description telling your
   team members what the change is for. Bonus points if you can link it to an issue or story.
6. Add reviewers, wait for approval, and make sure CI passes.
7. Depending on if your specific change, either:
  a. If this is a normal, run-of-the-mill change that we want to put upstream, add the
     `open-upstream` label to your PR and close your PR. In a few minutes, Actions will
     comment on your PR with a link that lets you open the same PR into upstream.
  b. If this is an urgent change that we want in `rocm-main` right now but also want upstream,
     add the `open-upstream` label, merge your PR, and then follow the link that 
  c. If this is a change that we only want to keep in `rocm/jax` and not push into upstream,
     squash and merge your PR.

If you submitted your PR upstream with `open-upstream`, you should see your change in `rocm-main`
the next time the `ROCm Nightly Upstream Sync` workflow is run and the PR that it creates is
merged.

## Daily Upstream Sync

Every day, GitHub Actions will attempt to run the `ROCm Nightly Upstream Sync` workflow. This job
normally does this on its own, but requires a developer to intervene if there's a merge conflict
or if the PR fails CI. Devs should fix or resolve issues with the merge by adding commits to the
PR's branch.

# Branches

 * `rocm-main` - the default "trunk" branch for this repo.  Should only be changed submitting PRs to it from feature branches created by devs.
 * `main` - a copy of `jax-ml/jax:main`. This branch is "read-only" and should only be changed by GitHub Actions.

# CI Workflows

We use GitHub Actions to run tests on PRs and to automate some of our
development tasks. These all live in `.github/workflows`.

| Name                       | File                             | Trigger                                              | Description                                                                            |
|----------------------------|----------------------------------|------------------------------------------------------|----------------------------------------------------------------------------------------|
| ROCm GPU CI                | `rocm-ci.yml`                    | Open or commit changes to a PR targeting `rocm-main` | Builds and runs JAX on ROCm for PRs going into `rocm-main`                             |
| ROCm Open Upstream PR      | `rocm-open-upstream-pr.yml`      | Add the `open-upstream` label to a PR                | Copies changes from a PR aimed at `rocm-main` into a new PR aimed at upstream's `main` |
| ROCm Nightly Upstream Sync | `rocm-nightly-upstream-sync.yml` | Runs nightly, can be triggered manually via Actions  | Opens a PR that merges changes from upstream `main` into our `rocm-main` branch        |


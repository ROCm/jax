#!/usr/bin/env bash

# run test module with multi-gpu reqirements. We currently do not have a way to filter tests.
# this issue is also tracked in https://github.com/google/jax/issues/7323
python3 -m pytest --reruns 3 -x tests/pmap_test.py
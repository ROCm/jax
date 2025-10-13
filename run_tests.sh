#!/bin/bash

./bazel-7.4.1-linux-x86_64 test \
	--subcommands \
	--config=rocm \
	--//jax:build_jaxlib=false \
	"//tests:core_test_gpu"


#!/bin/bash

./bazel-7.4.1-linux-x86_64 test \
	--subcommands \
	--config=rocm \
	--config=rocm_rbe \
	--//jax:build_jaxlib=false \
	"//tests:core_test_gpu"


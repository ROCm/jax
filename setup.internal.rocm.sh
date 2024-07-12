#!/usr/bin/env bash
#==============================================================================
#
# setup.internal.rocm.sh: Prepare the internal ROCM installation on the container.
# Usage: setup.internal.rocm.sh <ROCM_VERSION>
set -x

ROCM_MAJ_MIN=$1

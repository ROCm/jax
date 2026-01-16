# ROCm Pytest Workflow Porting Plan

## Overview
This document outlines the plan to port the CUDA pytest workflow (`.github/workflows/pytest_cuda.yml`) to support ROCm testing, creating `.github/workflows/pytest_rocm.yml` and associated test scripts.

## ⚠️ CRITICAL REQUIREMENT: Custom jaxlib for ROCm

**ROCm requires a custom-built jaxlib** - it cannot use the standard PyPI jaxlib. This is a fundamental difference from CUDA workflows and must be addressed in all implementation steps.

**Implications:**
- Custom jaxlib must be built and made available (via GitHub Actions artifacts or local build)
- Download action must support downloading custom jaxlib (not just PyPI)
- Workflow must download custom jaxlib before running tests
- Integration workflows must ensure custom jaxlib is built before pytest runs

## Current State Analysis

### CUDA Workflow Components
1. **Workflow File**: `.github/workflows/pytest_cuda.yml`
   - Uses `workflow_call` pattern (reusable workflow)
   - Downloads wheels via `download-jax-cuda-wheels` action
   - Sets PyPI extras based on CUDA version
   - Executes `ci/run_pytest_cuda.sh`
   - Supports multiple CUDA versions (12.1, 12.9, 13)
   - Uses CUDA-specific Docker containers

2. **Test Script**: `ci/run_pytest_cuda.sh`
   - Uses `nvidia-smi` for GPU detection
   - Calculates optimal parallel processes based on:
     - GPU count and memory
     - CPU core count
     - System RAM
   - Sets CUDA-specific environment variables:
     - `JAX_ENABLE_CUDA_XDIST`
     - `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
     - `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1`
   - Runs pytest with specific test exclusions

3. **Download Action**: `.github/actions/download-jax-cuda-wheels/action.yml`
   - Downloads from GCS or PyPI
   - Supports both "head" and "pypi_latest" versions
   - Handles CUDA version mapping

### Existing ROCm Infrastructure
1. **Download Action**: `.github/actions/download-jax-rocm-wheels/action.yml` ✅ EXISTS
   - Downloads from PyPI (currently only `pypi_latest` supported)
   - ROCm version mapping (6.x → "60")
   - Handles ROCm plugin wheels
   - ⚠️ **CRITICAL**: ROCm requires a custom jaxlib (cannot use PyPI jaxlib)
   - Currently only supports `pypi_latest` - needs enhancement to support custom jaxlib

2. **ROCm CI Workflow**: `.github/workflows/rocm-ci.yml` ✅ EXISTS
   - Uses Docker-based build and test
   - Runner: `linux-x86_64-cirrascale-64-8gpu-amd-mi250`
   - ROCm version: 6.3.3
   - Different approach (builds in Docker, tests in Docker)

3. **ROCm Build Scripts**: `build/rocm/` ✅ EXISTS
   - Build infrastructure available

## Porting Plan

### Phase 1: Create ROCm Test Script (`ci/run_pytest_rocm.sh`) ⚠️ PARTIALLY COMPLETED

**Status**: Core functionality implemented in commit 4b9488d98, needs completion for CI integration

**Implementation Details:**
1. **GPU Detection**: Uses `amd-smi list | grep "GPU:" | wc -l` for GPU count
2. **Memory Detection**: Uses `rocm-smi -d 0 --showmeminfo vram` for GPU memory
3. **Environment Variables**:
   - Sets both `JAX_ENABLE_CUDA_XDIST` and `JAX_ENABLE_ROCM_XDIST` to `$gpu_count`
   - Uses `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
   - Uses `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1`
4. **Parallel Process Calculation**: Same logic as CUDA (GPU capacity, CPU cores, RAM)
5. **Test Exclusions**: Same as CUDA (multi_device_test, multiprocess_gpu_test, compilation_cache_test)
6. **Test Reporting**: Adds CSV and HTML report generation

**Key Differences from CUDA:**
- Uses `amd-smi` for GPU count (instead of `nvidia-smi`)
- Uses `rocm-smi` for memory info (instead of `nvidia-smi`)
- Sets both CUDA and ROCm xdist variables (for compatibility)
- Generates test reports (CSV/HTML) and logs to file

**Missing from Implementation:**
- ⚠️ No shebang (`#!/bin/bash`)
- ⚠️ No copyright header
- ⚠️ No `set -exu -o history -o allexport` flags
- ⚠️ No sourcing of `ci/envs/default.env`
- ⚠️ No wheel installation step (`install_wheels_locally.sh`)
- ⚠️ No build environment setup (`setup_build_environment.sh`)
- ⚠️ No package listing or JAX device check
- ⚠️ Uses `pytest` directly instead of `$JAXCI_PYTHON -m pytest`

### Phase 2: Create ROCm Pytest Workflow (`.github/workflows/pytest_rocm.yml`)

**Dependencies:**
- `.github/actions/download-jax-rocm-wheels` (already exists)
- `ci/run_pytest_rocm.sh` (to be created in Phase 1)

**Workflow Structure:**
1. **Inputs** (similar to CUDA workflow):
   - `runner`: ROCm-capable runner (e.g., `linux-x86_64-cirrascale-64-8gpu-amd-mi250`)
   - `python`: Python version (default: "3.12")
   - `rocm-version`: ROCm version (default: "6.3.3")
   - `jaxlib-version`: Which jaxlib version? `"head"` (custom from artifacts), `"custom"`, or `"pypi_latest"` (default: "head")
   - `custom-jaxlib-artifact-name`: Name of GitHub Actions artifact containing custom jaxlib (required if jaxlib-version is "head" or "custom")
   - `use-rocm-pip-wheels`: Whether to use PyPI wheels for plugins (default: false)
   - `enable-x64`: Enable x64 mode (default: "0")
   - `skip-download-jaxlib-and-rocm-plugins`: Skip plugin downloads (default: '0')
   - `download-jax-from-pypi`: Whether to download JAX wheel from PyPI (default: '1')
   - `halt-for-connection`: Debug connection flag

2. **Container Selection**:
   - May need ROCm-specific container or use host system
   - Check if ROCm containers exist in `ml-public-container`
   - If not, may need to use host system or build custom container

3. **Steps**:
   - Checkout code
   - **Download custom jaxlib** (if jaxlib-version is "head" or "custom"):
     - Download from GitHub Actions artifacts (from build workflow)
     - Or download from local build directory
   - Download ROCm wheels using `download-jax-rocm-wheels` action
     - This will download JAX wheel (from PyPI) and ROCm plugins (from PyPI)
     - Custom jaxlib should already be in `dist/` from previous step
   - Set PyPI extras (if needed for ROCm)
   - Install Python dependencies
   - Wait for connection (debugging)
   - Run `ci/run_pytest_rocm.sh`

**Key Differences from CUDA:**
- No container selection logic (ROCm may run on host)
- Different PyPI extras handling (ROCm doesn't have version-specific extras like CUDA)
- Different runner requirements

### Phase 3: Integration with Wheel Test Workflows

**Files to Update:**
1. `.github/workflows/wheel_tests_continuous.yml`
   - Add `run-pytest-rocm` job
   - Add `build-rocm-artifacts` job (if needed)
   - Matrix strategy for ROCm versions

2. `.github/workflows/wheel_tests_nightly_release.yml`
   - Add `run-pytest-rocm` job
   - Matrix strategy for ROCm versions

**Considerations:**
- ROCm artifacts may need to be built separately or downloaded from PyPI
- Check if ROCm artifacts are built in `build_artifacts.yml` or separate workflow
- May need to add ROCm artifact build jobs

### Phase 4: Enhance Download Action for Custom jaxlib

**CRITICAL REQUIREMENT**: ROCm requires a custom-built jaxlib (cannot use PyPI jaxlib)

**Current State**: `.github/actions/download-jax-rocm-wheels/action.yml`
- Only supports `pypi_latest` for jaxlib-version
- Downloads from PyPI only (no GCS support for ROCm artifacts)
- **MUST BE UPDATED** to support custom jaxlib download

**Required Changes**:
1. **Add support for custom jaxlib download**:
   - Add `jaxlib-version` input with options: `"head"`, `"custom"`, `"pypi_latest"`
   - Add `custom-jaxlib-source` input to specify where custom jaxlib comes from:
     - GitHub Actions artifacts (from build workflow)
     - Local build directory
     - Other artifact storage (if available)
   - Add `custom-jaxlib-artifact-name` input (if using GitHub Actions artifacts)
   - Add `custom-jaxlib-path` input (if using local path)

2. **Download logic updates**:
   - When `jaxlib-version == "head"` or `"custom"`: Download custom jaxlib from specified source
   - When `jaxlib-version == "pypi_latest"`: Download from PyPI (for testing only, not recommended for ROCm)
   - Custom jaxlib should be downloaded to `$(pwd)/dist/` directory

3. **Artifact download options** (choose one or support multiple):
   - **Option A**: Download from GitHub Actions artifacts (if build workflow uploads them)
     ```yaml
     - uses: actions/download-artifact@v4
       with:
         name: ${{ inputs.custom-jaxlib-artifact-name }}
         path: $(pwd)/dist/
     ```
   - **Option B**: Download from local build directory (if built in same workflow)
   - **Option C**: Download from other artifact storage (if ROCm team has alternative)

**Note**: GCS support has been removed as ROCm team doesn't have GCS access. Custom jaxlib must come from GitHub Actions artifacts or local builds.

## Current Implementation Status

### `ci/run_pytest_rocm.sh` - Partially Complete (commit 4b9488d98)

**What's Implemented:**
- ✅ GPU count detection using `amd-smi list`
- ✅ GPU memory detection using `rocm-smi -d 0 --showmeminfo vram`
- ✅ Parallel process calculation (GPU capacity, CPU cores, RAM limits)
- ✅ Environment variables (JAX_ENABLE_CUDA_XDIST, JAX_ENABLE_ROCM_XDIST, XLA_FLAGS)
- ✅ Test exclusions (same as CUDA)
- ✅ CSV and HTML test report generation

**What's Missing (compared to `run_pytest_cuda.sh`):**
- ❌ Script header (shebang, copyright, license)
- ❌ Bash flags: `set -exu -o history -o allexport`
- ❌ Source `ci/envs/default.env`
- ❌ Wheel installation: `source ./ci/utilities/install_wheels_locally.sh`
- ❌ Package listing: `$JAXCI_PYTHON -m uv pip freeze`
- ❌ JAX device check: `$JAXCI_PYTHON -c "import jax; print(jax.default_backend())..."`
- ❌ ROCm system info: `rocm-smi` output (equivalent to `nvidia-smi`)
- ❌ Use `$JAXCI_PYTHON -m pytest` instead of `pytest`
- ❌ Test examples directory (CUDA script tests both `tests` and `examples`)

**Intentionally Skipped (not needed for PyPI wheel testing):**
- ✅ Build environment setup: `source ci/utilities/setup_build_environment.sh` (only needed for Bazel/XLA builds)

**Unique ROCm Features (not in CUDA):**
- ✅ CSV report generation (`--csv=tests-report.csv`)
- ✅ HTML report generation (`--html=tests-report.html --self-contained-html`)
- ✅ Log output to file (`2>&1 | tee jax_0.8.0_UT.log`)
- ✅ Uses both `amd-smi` and `rocm-smi` commands

## Detailed Implementation Steps

### Step 0: Complete `ci/run_pytest_rocm.sh` - Add Missing Components

**Priority 1 - Critical for functionality:**
1. Add script header:
   ```bash
   #!/bin/bash
   # Copyright 2024 The JAX Authors.
   # [License header...]
   set -exu -o history -o allexport
   ```

2. Source environment and install wheels:
   ```bash
   # Source default JAXCI environment variables.
   source ci/envs/default.env
   
   # Install jaxlib and ROCm plugin wheels
   echo "Installing wheels locally..."
   source ./ci/utilities/install_wheels_locally.sh
   ```

3. Replace `pytest` with `$JAXCI_PYTHON -m pytest`

**Priority 2 - Important for debugging:**
4. Add diagnostic output:
   ```bash
   # Print all the installed packages
   echo "Installed packages:"
   "$JAXCI_PYTHON" -m uv pip freeze
   
   "$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"
   
   rocm-smi
   ```

**Priority 3 - TODOs for future work:**
5. Add TODOs in the script:
   ```bash
   # TODO: Add examples directory to test suite
   # Currently only testing: tests
   # CUDA tests both: tests examples
   
   # TODO: Verify if CSV/HTML report generation should be kept
   # This is unique to ROCm implementation, not in CUDA
   
   # TODO: Verify if log file output should be kept
   # This is unique to ROCm implementation, not in CUDA
   
   # TODO: Consider adding setup_build_environment.sh if we need to support
   # non-PyPI workflows (e.g., building from source, using local XLA)
   ```

### Step 1: Update `ci/utilities/install_wheels_locally.sh`
**File**: `ci/utilities/install_wheels_locally.sh`

**Change Required**: Add ROCm wheel patterns to the find command

**Note**: This is required even for PyPI wheels because they are downloaded to `dist/` first, then the script finds and installs them.
```bash
# Current line 21:
WHEELS=( $(/usr/bin/find "$JAXCI_OUTPUT_DIR/" -type f \(  -name "*jax*py3*" -o -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*" \)) )

# Updated line 21:
WHEELS=( $(/usr/bin/find "$JAXCI_OUTPUT_DIR/" -type f \(  -name "*jax*py3*" -o -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*" -o -name "*jax*rocm*pjrt*" -o -name "*jax*rocm*plugin*" \)) )
```

**Rationale**: The script currently only finds CUDA wheels. Adding ROCm patterns ensures ROCm wheels are also installed.

### Step 2: Create `.github/workflows/pytest_rocm.yml`
**File**: `ci/run_pytest_rocm.sh`

**Current Implementation (commit 4b9488d98):**
```bash
# GPU detection using amd-smi and rocm-smi
export gpu_count=$(amd-smi list | grep "GPU:" | wc -l)
export memory_per_gpu_mib=$(rocm-smi -d 0 --showmeminfo vram | grep -i "vram total" | awk '{print int($NF/1024/1024)}' | head -1)

# Environment variables
export JAX_ENABLE_CUDA_XDIST="$gpu_count"
export JAX_ENABLE_ROCM_XDIST="$gpu_count"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1

# Parallel process calculation (same as CUDA)
# Test execution with CSV/HTML reporting
```

**Required Additions for First Implementation:**
1. Add script header (shebang, copyright, set flags)
2. Source `ci/envs/default.env`
3. Add wheel installation step (`source ./ci/utilities/install_wheels_locally.sh`)
4. Add package listing (`$JAXCI_PYTHON -m uv pip freeze`)
5. Add JAX device check (`$JAXCI_PYTHON -c "import jax; print(jax.default_backend()); print(jax.devices())"`)
6. Replace `pytest` with `$JAXCI_PYTHON -m pytest`
7. Add `rocm-smi` output (equivalent to `nvidia-smi` in CUDA script)

**TODOs to Add in Script (for future enhancements):**
- TODO: Add `examples` directory to test suite (currently only tests `tests/`)
- TODO: Consider adding build environment setup if needed for non-PyPI workflows
- TODO: Verify CSV/HTML report generation is desired (unique to ROCm implementation)
- TODO: Verify log file output is desired (unique to ROCm implementation)

### Step 2: Create `.github/workflows/pytest_rocm.yml`
**File**: `.github/workflows/pytest_rocm.yml`

**Structure:**
- Mirror `pytest_cuda.yml` structure
- Replace CUDA-specific inputs with ROCm equivalents
- **Add step to download custom jaxlib** (before download-jax-rocm-wheels action):
  ```yaml
  - name: Download custom jaxlib from artifacts
    if: ${{ inputs.jaxlib-version == 'head' || inputs.jaxlib-version == 'custom' }}
    uses: actions/download-artifact@v4
    with:
      name: ${{ inputs.custom-jaxlib-artifact-name }}
      path: $(pwd)/dist/
  ```
- Use `download-jax-rocm-wheels` action (will download JAX and ROCm plugins, custom jaxlib already in dist/)
- Call `ci/run_pytest_rocm.sh`
- Handle container selection (may be null for ROCm)

### Step 4: Test and Validate
**Files**: 
- `.github/workflows/wheel_tests_continuous.yml`
- `.github/workflows/wheel_tests_nightly_release.yml`

**Add:**
```yaml
build-rocm-jaxlib-artifact:
  # Build custom jaxlib for ROCm (if not already built)
  # This job should build and upload jaxlib as GitHub Actions artifact
  # OR use existing build workflow that produces ROCm jaxlib

run-pytest-rocm:
  needs: [build-jax-artifact, build-rocm-jaxlib-artifact]  # or build-jaxlib-artifact if shared
  uses: ./.github/workflows/pytest_rocm.yml
  strategy:
    matrix:
      runner: ["linux-x86_64-cirrascale-64-8gpu-amd-mi250"]
      python: ["3.11"]
      rocm-version: ["6.3.3"]
      enable-x64: [1, 0]
  with:
    runner: ${{ matrix.runner }}
    python: ${{ matrix.python }}
    rocm-version: ${{ matrix.rocm-version }}
    jaxlib-version: "head"  # Use custom jaxlib
    custom-jaxlib-artifact-name: "rocm-jaxlib-py${{ matrix.python }}-rocm${{ matrix.rocm-version }}"
    enable-x64: ${{ matrix.enable-x64 }}
    download-jax-from-pypi: "1"
```

### Step 5: Test and Validate
1. Test `rocm-smi` and `amd-smi` command availability and output format
2. Verify ROCm device enumeration
3. Test wheel installation process
4. Validate pytest execution with ROCm backend
5. Check test exclusions (may need ROCm-specific ones)
6. Verify custom jaxlib installation works correctly

## Dependencies Summary

### Existing (Ready to Use):
- ✅ `.github/actions/download-jax-rocm-wheels/action.yml`
- ✅ `ci/envs/default.env`
- ✅ `ci/utilities/install_wheels_locally.sh` (ROCm wheel patterns added)
- ✅ ROCm runner: `linux-x86_64-cirrascale-64-8gpu-amd-mi250`
- ✅ ROCm build infrastructure

### Not Required (for PyPI wheel testing):
- ✅ `ci/utilities/setup_build_environment.sh` (only needed for Bazel/XLA builds, not PyPI wheels)

### To Be Created:
- ✅ `ci/run_pytest_rocm.sh` (COMPLETED - cherry-picked from commit 4b9488d98)
- ❌ `.github/workflows/pytest_rocm.yml` (port from `pytest_cuda.yml`)

### To Be Updated:
- ⚠️ `.github/workflows/wheel_tests_continuous.yml` (add ROCm job)
- ⚠️ `.github/workflows/wheel_tests_nightly_release.yml` (add ROCm job)
- ⚠️ `.github/actions/download-jax-rocm-wheels/action.yml` (CRITICAL: Add custom jaxlib support - "head" and "custom" versions)
- ⚠️ `ci/run_pytest_rocm.sh` (add missing header, sourcing, and diagnostic output)

## Key Differences: CUDA vs ROCm

| Aspect | CUDA | ROCm |
|--------|------|------|
| GPU Detection | `nvidia-smi` | `rocm-smi` |
| Device Enumeration | NVIDIA GPUs | AMD GPUs |
| Container Support | CUDA containers available | May need host system |
| PyPI Extras | `cuda12`, `cuda12-local`, `cuda13` | Likely `rocm60` or similar |
| XLA Flags | CUDA-specific | ROCm-specific |
| Runner | Various NVIDIA GPU runners | `linux-x86_64-cirrascale-64-8gpu-amd-mi250` |
| Artifact Source | GCS (head) or PyPI | PyPI (plugins) + GitHub Actions artifacts (custom jaxlib) |
| Custom jaxlib | Downloaded from GCS | **REQUIRED** - Downloaded from GitHub Actions artifacts or local build |

## Open Questions

1. **Container Support**: Do ROCm containers exist in `ml-public-container`? Or should we use host system?
2. **ROCm Version Matrix**: What ROCm versions should be tested? (Currently 6.3.3)
3. **Custom jaxlib Source**: ~~Where is custom jaxlib stored?~~ **RESOLVED**: ROCm requires custom jaxlib. It should be downloaded from GitHub Actions artifacts (uploaded by build workflow) or from local build directory. Cannot use PyPI jaxlib.
4. **PyPI Extras**: What are the ROCm PyPI extra names? (e.g., `rocm60`, `rocm63`)
5. **XLA Flags**: What ROCm-specific XLA flags are needed?
6. **Test Exclusions**: Are there ROCm-specific tests that should be excluded?
7. **Parallel Execution**: Does ROCm support the same xdist mechanism as CUDA?

## Risk Assessment

### Low Risk:
- Workflow structure porting (straightforward)
- Download action integration (already exists)
- Basic script structure

### Medium Risk:
- `rocm-smi` command syntax and output parsing
- ROCm-specific environment variables
- Container/host system setup
- **Custom jaxlib download and integration** (critical for ROCm functionality)

### High Risk:
- ROCm device enumeration differences
- XLA flags and configuration
- Test compatibility and exclusions
- Performance/parallel execution differences

## Next Steps

1. **Research Phase**:
   - Test `rocm-smi` commands on ROCm runner
   - Verify ROCm device enumeration
   - Check ROCm-specific XLA configuration
   - Identify ROCm PyPI extra names

2. **Implementation Phase**:
   - **CRITICAL**: Update `.github/actions/download-jax-rocm-wheels/action.yml` to support custom jaxlib
   - ✅ Complete `ci/run_pytest_rocm.sh` (add missing header, sourcing, and setup steps)
   - Create `.github/workflows/pytest_rocm.yml` (with custom jaxlib download step)
   - Update wheel test workflows (ensure custom jaxlib is built/available)

3. **Testing Phase**:
   - Test workflow execution
   - Validate GPU detection
   - Verify test execution
   - Check parallel execution

4. **Integration Phase**:
   - Integrate with continuous workflows
   - Integrate with nightly/release workflows
   - Monitor and iterate

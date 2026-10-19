#!/usr/bin/env bash
# Debug branch only: build librocprofiler-sdk.so.1 from the rocm-systems commit that the
# TheRock 10.2.0a20261017 nightly pins, plus rocm-systems PR 13720, against that nightly's
# rocm-sdk-core/devel wheels, with TheRock's rocprofiler-sdk CMake options.
#
# usage: ci/rocm_build_rprof13720.sh OUTPUT_DIR
set -euxo pipefail

out=$(realpath -m "${1:?usage: $0 OUTPUT_DIR}")
nightly=10.2.0a20261017
rocm_systems_pin=2a8fffb7d64c59f83fe0d2efa22ed31d87d8adbf
pr_commit=8b33b3d15c5a6782f9db9e3639c616042fb20dbe
whl_index=https://nightly.repo.amd.com/rocm/whl-next

work=$(mktemp -d)
cd "$work"

# rocm-sdk-devel symlinks into ../../_rocm_sdk_core, so both trees share one parent.
curl -fsSL -o core.whl "$whl_index/rocm-sdk-core/rocm_sdk_core-${nightly}-py3-none-linux_x86_64.whl"
python3 -m zipfile -e core.whl sdk
rm core.whl
# zipfile drops mode bits; the compilers in the devel tree link to these.
find sdk/_rocm_sdk_core -type f \( -path '*/bin/*' -o -path '*/libexec/*' \) -exec chmod +x {} +
curl -fsSL -o devel.whl "$whl_index/rocm-sdk-devel/rocm_sdk_devel-${nightly}-py3-none-linux_x86_64.whl"
python3 -c 'import shutil, sys, zipfile; shutil.copyfileobj(zipfile.ZipFile(sys.argv[1]).open("rocm_sdk_devel/_devel.tar"), sys.stdout.buffer, 1 << 20)' devel.whl \
  | tar x -C sdk
rm devel.whl
ROCM=$work/sdk/_rocm_sdk_devel
orig_lib=$work/sdk/_rocm_sdk_core/lib/librocprofiler-sdk.so.1
md5sum "$orig_lib"

git init -q src
cd src
git remote add origin https://github.com/ROCm/rocm-systems.git
git sparse-checkout set --no-cone /projects/rocprofiler-sdk/ /shared/ /.gitmodules
git fetch -q --filter=blob:none --depth=1 origin "$rocm_systems_pin"
git fetch -q --filter=blob:none --depth=2 origin "$pr_commit"
git checkout -q "$rocm_systems_pin"
git cherry-pick --no-commit "$pr_commit"
git status --short
ext=projects/rocprofiler-sdk/external
git submodule update --init --depth 1 --jobs 8 \
  "$ext/abseil-cpp" "$ext/cereal" "$ext/elfio" "$ext/filesystem" "$ext/fmt" "$ext/gotcha" \
  "$ext/json" "$ext/perfetto" "$ext/ptl" "$ext/pybind11" "$ext/sqlite" "$ext/yaml-cpp"
cd "$work"

python3 -m venv venv
venv/bin/pip install -q "cmake<4" ninja pkgconf patchelf
export PATH=$work/venv/bin:$PATH
export PKG_CONFIG_PATH=$ROCM/lib/rocm_sysdeps/lib/pkgconfig:$ROCM/lib/pkgconfig

# rccl and MPI are not rocprofiler-sdk dependencies in TheRock, so the nightly uses the
# internal API-trace headers for them.
cmake -G Ninja -S src/projects/rocprofiler-sdk -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$ROCM;$ROCM/lib/rocm_sysdeps" \
  -DROCM_PATH="$ROCM" \
  -DHIP_PLATFORM=amd \
  -DROCPROFILER_BUILD_CI=ON \
  -DROCPROFILER_BUILD_DEVELOPER=OFF \
  -DROCPROFILER_BUILD_TESTS=OFF \
  -DROCPROFILER_BUILD_INTEGRATION_TESTS=OFF \
  -DROCPROFILER_BUILD_SAMPLES=OFF \
  -DROCPROFILER_BUILD_YAML_CPP=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_rccl=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON \
  -DROCPROFILER_PYTHON_VERSIONS="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
  -DROCPROFILER_PYTHON_EXECUTABLES="$(command -v python3)"
cmake --build build --target rocprofiler-sdk-shared-library -- -j"$(nproc)"

mkdir -p "$out"
new_lib=$out/librocprofiler-sdk.so.1
cp "$(readlink -f build/lib/librocprofiler-sdk.so.1)" "$new_lib"
patchelf --force-rpath --set-rpath "$(patchelf --print-rpath "$orig_lib")" "$new_lib"
strings "$new_lib" | grep -F 'cleared from the descriptor flags'
cmp -s <(nm -D --defined-only "$orig_lib" | awk '$3 ~ /^rocprofiler_/ {print $3}') \
  <(nm -D --defined-only "$new_lib" | awk '$3 ~ /^rocprofiler_/ {print $3}')
readelf -d "$new_lib" | grep -E 'NEEDED|SONAME|PATH'
{
  echo "rocm-sdk-core ${nightly}"
  echo "rocm-systems ${rocm_systems_pin} + cherry-pick ${pr_commit}"
  md5sum "$orig_lib" | sed 's/ .*/  nightly librocprofiler-sdk.so.1/'
  md5sum "$new_lib" | sed 's/ .*/  patched librocprofiler-sdk.so.1/'
} | tee "$out/BUILD_INFO"

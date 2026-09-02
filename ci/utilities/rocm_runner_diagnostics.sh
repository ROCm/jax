#!/bin/bash
# Quick resource snapshot for ARC runner + DinD + nested job container.
# Safe to run from the jax-base job container (docker.sock mounted) or runner shell.
set -uo pipefail

section() { echo ""; echo "======== $* ========"; }

section "Host / pod (from this shell)"
echo "hostname: $(hostname)"
echo "nproc: $(nproc 2>/dev/null || echo '?')"
if [ -r /sys/fs/cgroup/cpu.max ]; then
  echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
  echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
  echo "cgroup cpuset.cpus.effective: $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo n/a)"
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  echo "cgroup cpu quota/period: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)/$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
fi
echo "/dev/shm: $(df -h /dev/shm 2>/dev/null | tail -1 || echo n/a)"
free -h 2>/dev/null || true

section "podinfo (runner host paths, if mounted)"
for f in /etc/podinfo/gha-gpu-isolation-settings /etc/podinfo/gha-docker-cpu-flags /etc/podinfo/gha-render-devices; do
  if [ -r "$f" ]; then
    echo "--- $f ---"
    cat "$f"
  else
    echo "missing: $f"
  fi
done

section "Docker (nested job containers)"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker info 2>/dev/null | grep -E "CPUs|Total Memory|Docker Root Dir|Cgroup Driver|Cgroup Version" || true
  echo "--- docker ps ---"
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null || true
  echo "--- docker stats (one sample) ---"
  docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}' 2>/dev/null || true
  cid=$(docker ps -q | head -1)
  if [ -n "$cid" ]; then
    echo "--- first container HostConfig (cpus, shm, cpuset) ---"
    docker inspect "$cid" --format 'NanoCpus={{.HostConfig.NanoCpus}} CpuShares={{.HostConfig.CpuShares}} CpusetCpus={{.HostConfig.CpusetCpus}} ShmSize={{.HostConfig.ShmSize}} Memory={{.HostConfig.Memory}}' 2>/dev/null || true
  fi
else
  echo "docker not available from this shell"
fi

section "ROCm / JAX"
if command -v rocminfo >/dev/null 2>&1; then
  echo "GPU count (rocminfo): $(rocminfo 2>/dev/null | grep -c 'Device Type:[[:space:]]*GPU' || echo 0)"
fi
if [ -n "${JAXCI_PYTHON:-}" ] && command -v "$JAXCI_PYTHON" >/dev/null 2>&1; then
  "$JAXCI_PYTHON" -c "import jax; print('jax devices:', len(jax.devices()), list(jax.devices()))" 2>/dev/null || true
fi

section "Environment"
echo "ROCM_PYTEST_WORKERS_PER_GPU=${ROCM_PYTEST_WORKERS_PER_GPU:-}"
echo "RUNNER_CPU_COUNT=${RUNNER_CPU_COUNT:-}"
echo "INPUT_RUNNER=${INPUT_RUNNER:-}"

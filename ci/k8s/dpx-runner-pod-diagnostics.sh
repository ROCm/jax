#!/bin/bash
# Run from a machine with VPN + kubeconfig for do-atl1-amd-gpu-dra-dpx-cluster.
# Usage: ./ci/k8s/dpx-runner-pod-diagnostics.sh [pod-name]
set -euo pipefail

NS="${ARC_NAMESPACE:-arc-rocm-jax}"
LABEL="${RUNNER_LABEL:-amd-do-linux.jax.gpu.gfx950.4-dpx}"

pod="${1:-}"
if [ -z "$pod" ]; then
  pod=$(kubectl get pods -n "$NS" -l "actions.github.com/scale-set-name=${LABEL}" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
fi

if [ -z "$pod" ]; then
  echo "No running pod found in $NS for scale set $LABEL"
  kubectl get pods -n "$NS" -o wide | grep -E '4-dpx|NAME' || true
  exit 1
fi

echo "======== kubectl describe pod $pod ========"
kubectl describe pod -n "$NS" "$pod" | sed -n '/Containers:/,/Conditions:/p'

echo ""
echo "======== kubectl top pod ========"
kubectl top pod -n "$NS" "$pod" --containers 2>/dev/null || echo "(metrics-server unavailable)"

for ctr in dind runner; do
  echo ""
  echo "======== container: $ctr ========"
  kubectl exec -n "$NS" "$pod" -c "$ctr" -- sh -c '
    echo "nproc=$(nproc)"
    if [ -r /sys/fs/cgroup/cpu.max ]; then
      echo "cpu.max=$(cat /sys/fs/cgroup/cpu.max)"
      echo "memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
      echo "cpuset.cpus.effective=$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null)"
    fi
    df -h /dev/shm 2>/dev/null | tail -1
    free -h 2>/dev/null | head -2
  ' 2>/dev/null || echo "container $ctr not exec-able"
done

echo ""
echo "======== docker stats from runner ========"
kubectl exec -n "$NS" "$pod" -c runner -- sh -c '
  docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
  docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null || true
  cat /etc/podinfo/gha-docker-cpu-flags 2>/dev/null || echo "no gha-docker-cpu-flags"
' 2>/dev/null || true

echo ""
echo "======== dind logs (cpuset slice) ========"
kubectl logs -n "$NS" "$pod" -c dind --tail=20 2>/dev/null | grep -E 'INFO:|WARN:' || kubectl logs -n "$NS" "$pod" -c dind --tail=5

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export JAX_ENABLE_X64=1

PYTEST_ARGS=(-v --tb=short --no-header)

# Allow filtering via first argument, e.g.: ./run_mha_tests.sh fwd
if [[ $# -gt 0 ]]; then
    PYTEST_ARGS+=(-k "$1")
fi

TESTS=(
    # --- Batch forward ---
    test_batch_fwd_shape
    test_batch_fwd_accuracy

    # --- Batch backward ---
    test_batch_bwd_shape
    test_batch_bwd_accuracy

    # --- Dropout ---
    test_dropout_fwd
    test_dropout_bwd

    # --- Sliding window attention ---
    test_swa_fwd
    test_swa_bwd

    # --- Bias and ALiBi ---
    test_bias_fwd
    test_bias_bwd
    test_alibi_fwd
    test_alibi_causal

    # --- Return values ---
    test_return_lse
    test_return_attn_probs_with_dropout

    # --- Padded head dimensions ---
    test_padded_head_dim_fwd
    test_padded_head_dim_bwd

    # --- Deterministic ---
    test_deterministic_consistency
    test_deterministic_bwd

    # --- Variable length ---
    test_varlen_fwd
    test_varlen_bwd

    # --- Edge cases ---
    test_decode_sq1_fwd_bwd
    test_sq_gt_sk_nomask
    test_sq_gt_sk_causal
    test_large_batch
    test_single_head
    test_many_heads

    # --- Regressions ---
    test_v3_bwd_sq_gt_sk_causal
    test_1024_1023_causal
    test_mqa_gqa_bwd_routing
    test_varlen_large_sk_causal
    test_gfx950_1block_override
    test_swa_not_v3_bwd
    test_all_head_dims_bwd
)

passed=0
failed=0
crashed=0
skipped_list=()

echo "============================================"
echo " AITer MHA Test Suite"
echo " $(date)"
echo " GPU: $(rocminfo 2>/dev/null | grep -m1 'gfx' | awk '{print $NF}' || echo 'unknown')"
echo "============================================"
echo ""

for test_name in "${TESTS[@]}"; do
    # Skip if a filter was given and this test doesn't match
    if [[ $# -gt 0 ]] && [[ "$test_name" != *"$1"* ]]; then
        continue
    fi

    echo -n "Running $test_name ... "

    output=$(pytest tests/test_aiter_mha.py -k "$test_name" "${PYTEST_ARGS[@]}" 2>&1) || true
    exit_code=${PIPESTATUS[0]:-$?}

    # Count results from pytest output
    if echo "$output" | grep -qE "[0-9]+ (passed|xpassed)"; then
        p=0; for n in $(echo "$output" | grep -oP '\d+ (?:x?passed)' | grep -oP '\d+'); do p=$((p + n)); done
        f=$(echo "$output" | grep -oP '\d+ failed' | grep -oP '\d+' || echo 0)

        if [[ "$f" -gt 0 ]]; then
            echo "FAILED ($p passed, $f failed)"
            failed=$((failed + f))
            passed=$((passed + p))
            echo "$output" | grep -E "FAILED|AssertionError|Error" | head -5
            echo ""
        else
            echo "PASSED ($p passed)"
            passed=$((passed + p))
        fi
    elif echo "$output" | grep -q "no tests ran"; then
        echo "SKIPPED (no tests matched)"
        skipped_list+=("$test_name")
    else
        echo "CRASHED (process killed)"
        crashed=$((crashed + 1))
        echo "$output" | tail -5
        echo ""
    fi
done

echo ""
echo "============================================"
echo " RESULTS"
echo "============================================"
echo " Passed:  $passed"
echo " Failed:  $failed"
echo " Crashed: $crashed"
if [[ ${#skipped_list[@]} -gt 0 ]]; then
    echo " Skipped: ${#skipped_list[@]} (${skipped_list[*]})"
fi
echo "============================================"

if [[ $failed -gt 0 || $crashed -gt 0 ]]; then
    exit 1
fi

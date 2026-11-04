#!/usr/bin/env bash
# EXP-009 end-to-end orchestration (Mac host + phone adb).
# Usage:
#   export DEVICE=10.29.11.213:42629
#   ./scripts/run_e2e.sh all|dump|policy|apply|prefill|cleanup
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP_ROOT="$(cd "$ROOT/.." && pwd)"
PKG="${PKG:-$HOME/code/exp/shared/runtime/llama.cpp-hexagon-pkg}"
SRC_LLAMA="${SRC_LLAMA:-$HOME/code/exp/shared/runtime/llama.cpp}"
DEVICE="${DEVICE:-10.29.11.213:42629}"
ADB="${ADB:-adb -s $DEVICE}"
PHONE_LLAMA=/data/local/tmp/llama.cpp
PHONE_TMP=/data/local/tmp/exp009_kv
MODEL=/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf
MMPROJ=/data/local/tmp/models/mmproj-model-f16.gguf
RESULTS="$ROOT/results"
HOST_PULL="$RESULTS/phone_pull"
EXP008_PY="$EXP_ROOT/EXP-008-kv-keep-a-merge/scripts/kv_keep_merge.py"
PHASE="${1:-all}"

mkdir -p "$RESULTS" "$HOST_PULL"

log() { printf '[%s] %s\n' "$(date '+%F %T CST')" "$*"; }

need_device() {
  $ADB get-state >/dev/null
}

build_tool() {
  log "build kv_gui_e2e for android aarch64"
  local ndk_root
  ndk_root="$(ls -d "$HOME/Library/Android/sdk/ndk/"* 2>/dev/null | sort -V | tail -1)"
  if [[ -z "${ndk_root:-}" ]]; then
    echo "NDK not found" >&2; exit 1
  fi
  local cxx="$ndk_root/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang++"
  if [[ ! -x "$cxx" ]]; then
    cxx="$ndk_root/toolchains/llvm/prebuilt/"*"/bin/aarch64-linux-android28-clang++"
    cxx=$(ls $cxx 2>/dev/null | head -1)
  fi
  mkdir -p "$ROOT/tools/build"
  # Prefer packaging headers; fall back to source include + common
  local incs=(-I"$PKG/include" -I"$SRC_LLAMA/include" -I"$SRC_LLAMA/common" -I"$SRC_LLAMA/ggml/include")
  # common/*.cpp needed for common_params — too heavy. Instead compile a slim tool
  # that only uses libllama C API without common. If full tool needs common, link libllama-common.so.
  "$cxx" -std=c++17 -O2 -fPIC -pie \
    "${incs[@]}" \
    "$ROOT/tools/kv_gui_e2e_min.cpp" \
    -L"$PKG/lib" -lllama -lllama-common -lggml -lggml-base \
    -llog -ldl -lm \
    -Wl,-rpath,/data/local/tmp/llama.cpp/lib \
    -o "$ROOT/tools/build/kv_gui_e2e"
  log "built $ROOT/tools/build/kv_gui_e2e"
}

push_tool() {
  need_device
  $ADB push "$ROOT/tools/build/kv_gui_e2e" "$PHONE_LLAMA/bin/kv_gui_e2e"
  $ADB shell "chmod 755 $PHONE_LLAMA/bin/kv_gui_e2e; mkdir -p $PHONE_TMP"
}

run_dump() {
  need_device
  log "phone dump phase (HTP0)"
  $ADB shell "rm -rf $PHONE_TMP; mkdir -p $PHONE_TMP"
  $ADB shell "cd $PHONE_LLAMA; \
    export LD_LIBRARY_PATH=$PHONE_LLAMA/lib ADSP_LIBRARY_PATH=$PHONE_LLAMA/lib GGML_HEXAGON_EXPERIMENTAL=1; \
    ./bin/kv_gui_e2e --phase dump --dump-dir $PHONE_TMP \
      -m $MODEL --no-mmap --device HTP0 -ngl 99 -c 2048 -n 8 -t 4" \
    | tee "$RESULTS/dump_phone.log"
  $ADB pull "$PHONE_TMP/segments.json" "$HOST_PULL/segments.json"
  $ADB pull "$PHONE_TMP/kv_seq0.bin" "$HOST_PULL/kv_seq0.bin"
  $ADB pull "$PHONE_TMP/kv_state.bin" "$HOST_PULL/kv_state.bin" || true
  ls -la "$HOST_PULL"
}

run_policy() {
  log "parse seq + EXP-008 policy → keep mask"
  conda run -n visiontrim-qwen25vl python "$ROOT/scripts/parse_kv_seq.py" \
    --seq "$HOST_PULL/kv_seq0.bin" \
    --meta "$HOST_PULL/segments.json" \
    --out-npz "$RESULTS/real_kv_segments.npz" \
    --feat-dim 64 | tee "$RESULTS/parse_log.json"
  conda run -n visiontrim-qwen25vl python "$ROOT/scripts/policy_to_keep_mask.py" \
    --npz "$RESULTS/real_kv_segments.npz" \
    --meta "$HOST_PULL/segments.json" \
    --exp008 "$EXP008_PY" \
    --out-mask "$RESULTS/keep_mask.txt" \
    --out-json "$RESULTS/policy_summary.json" | tee "$RESULTS/policy_log.json"
}

run_apply() {
  need_device
  $ADB push "$RESULTS/keep_mask.txt" "$PHONE_TMP/keep_mask.txt"
  log "phone apply keep mask + save compressed state"
  $ADB shell "cd $PHONE_LLAMA; \
    export LD_LIBRARY_PATH=$PHONE_LLAMA/lib ADSP_LIBRARY_PATH=$PHONE_LLAMA/lib GGML_HEXAGON_EXPERIMENTAL=1; \
    ./bin/kv_gui_e2e --phase apply --dump-dir $PHONE_TMP --keep-mask $PHONE_TMP/keep_mask.txt \
      -m $MODEL --no-mmap --device HTP0 -ngl 99 -c 2048 -n 8 -t 4" \
    | tee "$RESULTS/apply_phone.log"
  $ADB pull "$PHONE_TMP/apply_stats.json" "$RESULTS/apply_stats.json" || true
  $ADB pull "$PHONE_TMP/kv_seq0_compressed.bin" "$HOST_PULL/kv_seq0_compressed.bin" || true
}

run_prefill() {
  need_device
  # optional image
  local img_args=()
  local shot="$EXP_ROOT/EXP-003-phone-server/inputs/shot0.png"
  if [[ -f "$shot" ]]; then
    $ADB push "$shot" /data/local/tmp/shot0_exp009.png
  fi
  log "phone HTP0 prefill next prompt on compressed KV"
  $ADB shell "cd $PHONE_LLAMA; \
    export LD_LIBRARY_PATH=$PHONE_LLAMA/lib ADSP_LIBRARY_PATH=$PHONE_LLAMA/lib GGML_HEXAGON_EXPERIMENTAL=1; \
    ./bin/kv_gui_e2e --phase prefill --dump-dir $PHONE_TMP \
      -m $MODEL --no-mmap --device HTP0 -ngl 99 -c 2048 -n 24 -t 4 \
      --next-prompt 'Next screenshot observation: Wi-Fi switch is on the screen. Reply with the next GUI action briefly.'" \
    | tee "$RESULTS/prefill_phone.log"
  $ADB pull "$PHONE_TMP/prefill_stats.json" "$RESULTS/prefill_stats.json" || true
  $ADB pull "$PHONE_TMP/generation.txt" "$RESULTS/generation.txt" || true
}

cleanup() {
  need_device || true
  $ADB shell "rm -rf $PHONE_TMP /data/local/tmp/shot0_exp009.png; rm -f $PHONE_LLAMA/bin/kv_gui_e2e" || true
  log "cleaned phone temp"
}

case "$PHASE" in
  build) build_tool ;;
  push) build_tool; push_tool ;;
  dump) run_dump ;;
  policy) run_policy ;;
  apply) run_apply ;;
  prefill) run_prefill ;;
  cleanup) cleanup ;;
  all)
    build_tool
    push_tool
    run_dump
    run_policy
    run_apply
    run_prefill
    cleanup
    ;;
  *) echo "unknown phase $PHASE"; exit 1 ;;
esac

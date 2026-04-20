#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
DATASET="$ROOT_DIR/training_data/datasets/names.txt"
FIXTURE_DIR="$ROOT_DIR/training_data/fixtures/small_case"
PYTHON_BIN="${PYTHON:-python3}"

serial_python_valid=0
serial_valid=0
parallel_valid=0
baseline_small=""
baseline_medium=""
baseline_large=""

set_valid() {
    case "$1" in
        serial_python) serial_python_valid="$2" ;;
        serial_cpp) serial_valid="$2" ;;
        parallel_cpp) parallel_valid="$2" ;;
    esac
}

is_valid() {
    case "$1" in
        serial_python) [ "$serial_python_valid" -eq 1 ] ;;
        serial_cpp) [ "$serial_valid" -eq 1 ] ;;
        parallel_cpp) [ "$parallel_valid" -eq 1 ] ;;
        *) return 1 ;;
    esac
}

set_baseline() {
    case "$1" in
        small) baseline_small="$2" ;;
        medium) baseline_medium="$2" ;;
        large) baseline_large="$2" ;;
    esac
}

get_baseline() {
    case "$1" in
        small) printf "%s" "$baseline_small" ;;
        medium) printf "%s" "$baseline_medium" ;;
        large) printf "%s" "$baseline_large" ;;
    esac
}

run_validate_method() {
    local method="$1"
    case "$method" in
        serial_python)
            "$PYTHON_BIN" "$ROOT_DIR/methods/serial_python/serial.py" --mode validate --fixture-dir "$FIXTURE_DIR"
            ;;
        serial_cpp|parallel_cpp)
            "$BUILD_DIR/$method" --mode validate --fixture-dir "$FIXTURE_DIR"
            ;;
        *)
            return 1
            ;;
    esac
}

benchmark_once() {
    local method="$1"
    local preset="$2"
    local time_file
    time_file="$(mktemp)"
    case "$method" in
        serial_python)
            if ! /usr/bin/time -p "$PYTHON_BIN" "$ROOT_DIR/methods/serial_python/serial.py" \
                --mode benchmark \
                --dataset "$DATASET" \
                --preset "$preset" \
                > /dev/null 2> "$time_file"; then
                rm -f "$time_file"
                return 1
            fi
            ;;
        serial_cpp|parallel_cpp)
            if ! /usr/bin/time -p "$BUILD_DIR/$method" \
                --mode benchmark \
                --dataset "$DATASET" \
                --preset "$preset" \
                > /dev/null 2> "$time_file"; then
                rm -f "$time_file"
                return 1
            fi
            ;;
        *)
            rm -f "$time_file"
            return 1
            ;;
    esac
    awk '/^real / { print $2 }' "$time_file"
    rm -f "$time_file"
    return 0
}

printf "Building binaries and regenerating fixtures...\n"
if ! command -v nvcc >/dev/null 2>&1; then
    printf "parallel_cpp now requires nvcc on PATH. Install the CUDA toolkit before running the full benchmark sweep.\n" >&2
    exit 1
fi
if ! make -C "$ROOT_DIR" fixtures all; then
    printf "Build failed.\n" >&2
    exit 1
fi

for method in serial_python serial_cpp parallel_cpp; do
    printf "Validating %s...\n" "$method"
    if run_validate_method "$method"; then
        set_valid "$method" 1
    else
        printf "%s FAILED validation and will be skipped.\n" "$method"
        set_valid "$method" 0
    fi
done

for preset in small medium large; do
    printf "\nPreset: %s\n" "$preset"
    for method in serial_cpp serial_python parallel_cpp; do
        if ! is_valid "$method"; then
            printf "%s: INVALID\n" "$method"
            continue
        fi
        raw_time="$(benchmark_once "$method" "$preset")" || {
            printf "%s: benchmark failed\n" "$method"
            continue
        }
        printf "%s raw_real=%s\n" "$method" "$raw_time"
        if [ "$method" = "serial_cpp" ]; then
            set_baseline "$preset" "$raw_time"
            printf "%s speedup=1.000000\n" "$method"
        else
            baseline_value="$(get_baseline "$preset")"
            if [ -n "$baseline_value" ]; then
                speedup="$(awk -v base="$baseline_value" -v current="$raw_time" 'BEGIN { printf "%.6f", base / current }')"
                printf "%s speedup=%s\n" "$method" "$speedup"
            else
                printf "%s speedup=N/A\n" "$method"
            fi
        fi
    done
done

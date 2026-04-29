#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
DATASET = ROOT_DIR / "training_data" / "datasets" / "names.txt"
FIXTURE_DIR = ROOT_DIR / "training_data" / "fixtures" / "small_case"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark_results.json"
METHODS = ("serial_python", "serial_cpp", "parallel_cpp")
PRESETS = ("small", "medium", "large")


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def run_command(command: list[str]) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_key_value_line(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in reversed(text.splitlines()):
        if "=" not in line:
            continue
        for part in line.strip().split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        if fields:
            break
    return fields


def run_validate_method(method: str) -> CommandResult:
    if method == "serial_python":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_python" / "serial.py"),
                "--mode",
                "validate",
                "--fixture-dir",
                str(FIXTURE_DIR),
            ]
        )
    return run_command(
        [
            str(BUILD_DIR / method),
            "--mode",
            "validate",
            "--fixture-dir",
            str(FIXTURE_DIR),
        ]
    )


def run_benchmark_method(method: str, preset: str) -> CommandResult:
    if method == "serial_python":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_python" / "serial.py"),
                "--mode",
                "benchmark",
                "--dataset",
                str(DATASET),
                "--preset",
                preset,
            ]
        )
    return run_command(
        [
            str(BUILD_DIR / method),
            "--mode",
            "benchmark",
            "--dataset",
            str(DATASET),
            "--preset",
            preset,
        ]
    )


def main() -> int:
    output_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT

    results: dict[str, object] = {
        "dataset": str(DATASET),
        "fixture_dir": str(FIXTURE_DIR),
        "build": {},
        "validate": {},
        "benchmarks": [],
    }

    build_commands: list[list[str]] = [
        ["make", "-C", str(ROOT_DIR), "fixtures", "build/serial_cpp"],
    ]
    if shutil.which("nvcc") is not None:
        build_commands[0].append("build/parallel_cpp")
    else:
        results["build"] = {
            "parallel_cpp": {
                "status": "skipped",
                "reason": "nvcc not found on PATH",
            }
        }

    for command in build_commands:
        build_result = run_command(command)
        results["build"]["core"] = {
            "command": command,
            "returncode": build_result.returncode,
            "stdout": build_result.stdout,
            "stderr": build_result.stderr,
        }
        if build_result.returncode != 0:
            output_path.write_text(json.dumps(results, indent=2) + "\n")
            return build_result.returncode

    valid_methods: dict[str, bool] = {}
    for method in METHODS:
        if method == "parallel_cpp" and shutil.which("nvcc") is None:
            valid_methods[method] = False
            results["validate"][method] = {
                "status": "skipped",
                "reason": "nvcc not found on PATH",
            }
            continue

        validate_result = run_validate_method(method)
        valid_methods[method] = validate_result.returncode == 0
        results["validate"][method] = {
            "status": "pass" if validate_result.returncode == 0 else "fail",
            "returncode": validate_result.returncode,
            "stdout": validate_result.stdout,
            "stderr": validate_result.stderr,
        }

    for preset in PRESETS:
        for method in METHODS:
            if not valid_methods.get(method, False):
                results["benchmarks"].append(
                    {
                        "method": method,
                        "preset": preset,
                        "status": "skipped",
                    }
                )
                continue

            benchmark_result = run_benchmark_method(method, preset)
            parsed = parse_key_value_line(benchmark_result.stdout)
            results["benchmarks"].append(
                {
                    "method": method,
                    "preset": preset,
                    "status": "pass" if benchmark_result.returncode == 0 else "fail",
                    "returncode": benchmark_result.returncode,
                    "stdout": benchmark_result.stdout,
                    "stderr": benchmark_result.stderr,
                    "parsed": parsed,
                }
            )

    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote benchmark results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

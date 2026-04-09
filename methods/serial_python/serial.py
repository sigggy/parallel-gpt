"""
Thin Python runner for fixture generation, validation, and benchmarking.
The model math lives in kernel.py.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from kernel import ModelConfig, flatten_param_values, init_model, run_forward_backward


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "training_data" / "datasets" / "names.txt"
DEFAULT_FIXTURE_DIR = REPO_ROOT / "training_data" / "fixtures" / "small_case"
DEFAULT_SAMPLE_NAME = "anna"
DEFAULT_SEED = 42
VALIDATION_EPSILON = 1e-4
BENCHMARK_PRESETS = {
    "small": {"config": ModelConfig(n_layer=1, n_embd=64, block_size=64, n_head=4), "steps": 200},
    "medium": {"config": ModelConfig(n_layer=2, n_embd=128, block_size=64, n_head=8), "steps": 200},
    "large": {"config": ModelConfig(n_layer=4, n_embd=256, block_size=128, n_head=8), "steps": 100},
}


def load_docs(dataset_path: Path):
    with open(dataset_path) as handle:
        return [line.strip() for line in handle if line.strip()]


def build_vocab(docs):
    uchars = sorted(set("".join(docs)))
    bos = len(uchars)
    vocab = {ch: idx for idx, ch in enumerate(uchars)}
    return uchars, vocab, bos


def encode_doc(doc, vocab, bos):
    return [bos] + [vocab[ch] for ch in doc] + [bos]


def load_tokens(dataset_path: Path, sample_name: str):
    docs = load_docs(dataset_path)
    uchars, vocab, bos = build_vocab(docs)
    if any(ch not in vocab for ch in sample_name):
        raise ValueError(f"sample name contains characters not found in dataset: {sample_name}")
    tokens = encode_doc(sample_name, vocab, bos)
    return uchars, tokens


def pack_f32(values):
    return struct.pack("<" + ("f" * len(values)), *values)


def read_f32_file(path: Path):
    data = path.read_bytes()
    return list(struct.unpack("<" + ("f" * (len(data) // 4)), data))


def write_f32_file(path: Path, values):
    path.write_bytes(pack_f32(list(values)))


def load_model_from_f32(state_dict, config, values):
    cursor = 0
    for name in ("wte", "wpe", "lm_head"):
        matrix = state_dict[name]
        for row in matrix:
            for idx in range(len(row)):
                row[idx].data = values[cursor]
                cursor += 1
    for layer_idx in range(config.n_layer):
        for suffix in ("attn_wq", "attn_wk", "attn_wv", "attn_wo", "mlp_fc1", "mlp_fc2"):
            matrix = state_dict[f"layer{layer_idx}.{suffix}"]
            for row in matrix:
                for idx in range(len(row)):
                    row[idx].data = values[cursor]
                    cursor += 1
    if cursor != len(values):
        raise ValueError("fixture weights contain extra values")


def parse_manifest(manifest_path: Path):
    manifest = {}
    for line in manifest_path.read_text().splitlines():
        if not line:
            continue
        key, value = line.split("=", 1)
        manifest[key] = value
    return manifest


def parse_int_list(text: str):
    return [int(value) for value in text.split(",") if value]


def compare_arrays(label: str, actual, expected, epsilon: float):
    if len(actual) != len(expected):
        raise ValueError(f"{label} size mismatch")
    max_abs_error = 0.0
    max_idx = 0
    for idx, (left, right) in enumerate(zip(actual, expected)):
        abs_error = abs(left - right)
        if abs_error > max_abs_error:
            max_abs_error = abs_error
            max_idx = idx
    print(f"{label} max_abs_error={max_abs_error:.8g} at_index={max_idx}")
    if max_abs_error > epsilon:
        raise ValueError(f"{label} exceeded validation epsilon")


def parse_args():
    parser = argparse.ArgumentParser(description="serial Python benchmark scaffold")
    parser.add_argument("--mode", choices=["dump-fixtures", "validate", "benchmark"], required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--sample-name", default=DEFAULT_SAMPLE_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--preset", choices=sorted(BENCHMARK_PRESETS), default="small")
    parser.add_argument("--num-steps", type=int, default=None)
    return parser.parse_args()


def dump_fixtures(dataset_path: Path, fixture_dir: Path, sample_name: str, seed: int):
    uchars, tokens = load_tokens(dataset_path, sample_name)
    config = ModelConfig()
    state_dict, params = init_model(config, len(uchars) + 1, seed)
    result = run_forward_backward(tokens, state_dict, params, config)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "format_version=1",
        f"dataset_path={dataset_path}",
        f"sample_name={sample_name}",
        f"token_ids={','.join(str(token_id) for token_id in tokens)}",
        f"vocab_size={len(uchars) + 1}",
        f"n_layer={config.n_layer}",
        f"n_embd={config.n_embd}",
        f"block_size={config.block_size}",
        f"n_head={config.n_head}",
        f"validation_epsilon={VALIDATION_EPSILON}",
        "weights_init_file=weights_init.bin",
        "expected_logits_file=expected_logits.bin",
        "expected_loss_file=expected_loss.bin",
        "expected_grads_file=expected_grads.bin",
    ]

    (fixture_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")
    write_f32_file(fixture_dir / "weights_init.bin", flatten_param_values(state_dict, config))
    write_f32_file(fixture_dir / "expected_logits.bin", [value for row in result["logits"] for value in row])
    write_f32_file(fixture_dir / "expected_loss.bin", [result["loss"]])
    write_f32_file(fixture_dir / "expected_grads.bin", result["grads"])
    print(f"wrote fixture bundle to {fixture_dir}")


def validate_fixture(fixture_dir: Path, seed: int):
    manifest = parse_manifest(fixture_dir / "manifest.txt")
    config = ModelConfig(
        n_layer=int(manifest["n_layer"]),
        n_embd=int(manifest["n_embd"]),
        block_size=int(manifest["block_size"]),
        n_head=int(manifest["n_head"]),
    )
    vocab_size = int(manifest["vocab_size"])
    tokens = parse_int_list(manifest["token_ids"])
    epsilon = float(manifest["validation_epsilon"])

    state_dict, params = init_model(config, vocab_size, seed)
    load_model_from_f32(state_dict, config, read_f32_file(fixture_dir / manifest["weights_init_file"]))
    result = run_forward_backward(tokens, state_dict, params, config)

    compare_arrays("logits", [value for row in result["logits"] for value in row], read_f32_file(fixture_dir / manifest["expected_logits_file"]), epsilon)
    compare_arrays("loss", [result["loss"]], read_f32_file(fixture_dir / manifest["expected_loss_file"]), epsilon)
    compare_arrays("grads", result["grads"], read_f32_file(fixture_dir / manifest["expected_grads_file"]), epsilon)
    print("validation=pass")


def run_benchmark(dataset_path: Path, sample_name: str, seed: int, preset_name: str, num_steps: int | None):
    uchars, tokens = load_tokens(dataset_path, sample_name)
    preset = BENCHMARK_PRESETS[preset_name]
    config = preset["config"]
    steps = num_steps if num_steps is not None else preset["steps"]
    state_dict, params = init_model(config, len(uchars) + 1, seed)
    last_result = None
    for _ in range(steps):
        last_result = run_forward_backward(tokens, state_dict, params, config)
    print(
        "mode=benchmark "
        f"preset={preset_name} "
        f"steps={steps} "
        f"sample_name={sample_name} "
        f"loss={last_result['loss']:.6f}"
    )


def main():
    args = parse_args()
    if args.mode == "dump-fixtures":
        dump_fixtures(args.dataset, args.fixture_dir, args.sample_name, args.seed)
        return
    if args.mode == "validate":
        validate_fixture(args.fixture_dir, args.seed)
        return
    run_benchmark(args.dataset, args.sample_name, args.seed, args.preset, args.num_steps)


if __name__ == "__main__":
    main()

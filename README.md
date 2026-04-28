# Parallel GPT Project

## Commands

### Build everything

```bash
make all
```

Note: `parallel_cpp` is now a CUDA-only target. `make all` requires `nvcc` on `PATH`.

### Regenerate the validation fixture bundle

```bash
make fixtures
```

Or directly:

```bash
python3 methods/serial_python/serial.py \
  --mode dump-fixtures \
  --dataset training_data/datasets/names.txt \
  --fixture-dir training_data/fixtures/small_case \
  --sample-name anna
```

### Validate each method

Python reference:

```bash
python3 methods/serial_python/serial.py \
  --mode validate \
  --fixture-dir training_data/fixtures/small_case
```

Serial C++:

```bash
build/serial_cpp --mode validate --fixture-dir training_data/fixtures/small_case
```

Parallel C++:

```bash
build/parallel_cpp --mode validate --fixture-dir training_data/fixtures/small_case
```

Note: `parallel_cpp` is an outline scaffold and requires a CUDA build. Validation will fail until you implement the CUDA compute path in `methods/parallel_cpp/kernel.cu`.

### Benchmark a single method

Python reference:

```bash
python3 methods/serial_python/serial.py \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --preset small
```

Serial C++:

```bash
build/serial_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --preset small
```

Parallel C++:

```bash
build/parallel_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --preset small
```

Note: this binary currently demonstrates the host-side path and CUDA launch outline only. It will not produce a valid benchmark result until the compute kernels are implemented.

Optional presets:

- `small`
- `medium`
- `large`

Override how many names are processed if needed:

```bash
build/serial_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --preset small \
  --num-steps 10
```

### Run the full benchmark sweep

This rebuilds, regenerates fixtures, validates all methods, then times each valid method once.

```bash
bash scripts/run_benchmarks.sh
```

## Repo Layout

- `methods/serial_python/kernel.py`: Python reference forward kernel.
- `methods/serial_python/serial.py`: Python runner for fixture generation, validation, and benchmarking.
- `methods/serial_cpp/kernel.cpp`: Serial C++ forward kernel.
- `methods/serial_cpp/utils.cpp`: Serial C++ model setup and serialization helpers kept separate from kernel math.
- `methods/serial_cpp/main.cpp`: Serial C++ runner.
- `methods/parallel_cpp/kernel.cu`: CUDA-target translation unit showing how to allocate buffers and launch placeholder kernels.
- `methods/parallel_cpp/utils.cpp`: Parallel C++ model setup and serialization helpers kept separate from CUDA-specific code.
- `methods/parallel_cpp/main.cpp`: Parallel method runner.
- `training_data/datasets/names.txt`: dataset used for benchmarks.
- `training_data/fixtures/small_case/`: deterministic validation data generated from Python.

## How It Works

The project is built around one narrow workload: repeated forward passes on tokenized name data with next-token loss. The point is to compare implementations of the same kernel, not to build a full training framework.

Each method has the same shape:

- a thin runner file for CLI, dataset loading, and validation/benchmark orchestration
- a kernel file containing the actual GPT forward implementation

The Python version is the reference implementation. It is used to generate deterministic ground-truth files for:

- initial weights
- expected logits
- expected scalar loss

Those files live in `training_data/fixtures/small_case/`. Validation works by loading the fixture weights, running the method’s forward pass on the fixed token sequence from the manifest, and comparing the outputs against the Python ground truth within an epsilon.

Benchmarking keeps the GPT-style data flow without the training update. Each executable loads the dataset, builds the vocabulary, initializes weights from the fixed seed, and then processes the first `k` names in dataset order, where `k` is the preset size or `--num-steps`. For each benchmark step it builds `[BOS] + doc + [BOS]`, runs the forward pass across that sequence, and computes the next-token loss. The outer script uses `/usr/bin/time -p` and reports the raw wall-clock `real` time from one run. This means timing includes process startup and dataset loading for every method equally.

The current methods are:

- `serial_python`: correctness reference and optional timing reference
- `serial_cpp`: CPU baseline
- `parallel_cpp`: CUDA-target scaffold

Right now `parallel_cpp` keeps a copied host-side flow similar to `serial_cpp` up to the point where actual forward computation begins. After that boundary, the parallel method switches to a CUDA outline. `parallel_cpp` now builds only from the `.cu` translation unit and requires `nvcc`. The CUDA path is intentionally incomplete: it shows buffer upload, workspace allocation, placeholder kernel definitions, and example launch sites without implementing the transformer math for you.

# Parallel GPT Project

## Commands

### Build everything

```bash
make all
```

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

### Benchmark a single method

Python reference:

```bash
python3 methods/serial_python/serial.py \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --sample-name anna \
  --preset small
```

Serial C++:

```bash
build/serial_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --sample-name anna \
  --preset small
```

Parallel C++:

```bash
build/parallel_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --sample-name anna \
  --preset small
```

Optional presets:

- `small`
- `medium`
- `large`

Override the repeat count if needed:

```bash
build/serial_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --sample-name anna \
  --preset small \
  --num-steps 10
```

### Run the full benchmark sweep

This rebuilds, regenerates fixtures, validates all methods, then times each valid method.

```bash
bash scripts/run_benchmarks.sh
```

## Repo Layout

- `methods/serial_python/kernel.py`: Python reference forward/backward kernel.
- `methods/serial_python/serial.py`: Python runner for fixture generation, validation, and benchmarking.
- `methods/serial_cpp/kernel.cpp`: Serial C++ forward/backward kernel.
- `methods/serial_cpp/main.cpp`: Serial C++ runner.
- `methods/parallel_cpp/kernel.cpp`: Parallel method kernel. Right now this is the same as serial C++.
- `methods/parallel_cpp/main.cpp`: Parallel method runner.
- `training_data/datasets/names.txt`: dataset used for benchmarks.
- `training_data/fixtures/small_case/`: deterministic validation data generated from Python.

## How It Works

The project is built around one narrow workload: repeated forward pass plus backward pass on tokenized name data. The point is to compare implementations of the same kernel, not to build a full training framework.

Each method has the same shape:

- a thin runner file for CLI, dataset loading, and validation/benchmark orchestration
- a kernel file containing the actual GPT forward/backward implementation

The Python version is the reference implementation. It is used to generate deterministic ground-truth files for:

- initial weights
- expected logits
- expected scalar loss
- expected gradients

Those files live in `training_data/fixtures/small_case/`. Validation works by loading the fixture weights, running the method’s forward/backward pass on the fixed token sequence from the manifest, and comparing the outputs against the Python ground truth within an epsilon.

Benchmarking is intentionally simple. Each executable loads the dataset, builds the vocabulary, tokenizes the chosen sample name, initializes weights from the fixed seed, and then repeatedly runs the forward/backward kernel. The outer script uses `/usr/bin/time -p` and reports median wall-clock time across five runs. This means timing includes process startup and dataset loading for every method equally.

The current methods are:

- `serial_python`: correctness reference and optional timing reference
- `serial_cpp`: CPU baseline
- `parallel_cpp`: placeholder for the future CUDA implementation

Right now `parallel_cpp` is just a copy of `serial_cpp`. The idea is that CUDA work should happen mainly inside the `parallel_cpp` kernel file, while the validation and benchmark harness stay stable.

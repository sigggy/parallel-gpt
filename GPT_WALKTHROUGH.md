# GPT Walkthrough: Python Reference and Serial C++

This project is a small GPT-style next-token model built around one specific workload: take a tokenized name, run a forward pass, compute next-token loss, and run a backward pass. It is not a full training framework with optimizer updates, checkpoints, batching, or sampling loops. The Python version is the reference implementation, and the serial C++ version mirrors the same model in a more explicit systems-style form.

The sections below walk through the model in the same order the code executes. For each section:

1. what the section is doing conceptually
2. the Python code
3. the corresponding C++ code

For this repository, the dataset vocabulary is the 26 lowercase letters `a-z`, plus one special beginning-of-sequence token. That means the effective vocabulary size is `27`.

This walkthrough also uses one running example: the word `anna`.

- character IDs: `a = 0`, `n = 13`
- special token: `BOS = 26`
- tokenized `anna`: `[26, 0, 13, 13, 0, 26]`

That sequence gives the model five next-token prediction tasks:

- from `26` predict `0`
- from `0` predict `13`
- from `13` predict `13`
- from `13` predict `0`
- from `0` predict `26`

## 1. Runner and Workload Shape

At the top level, both implementations are thin runners. Their job is to choose a mode, load input data, build or load model weights, and then call the GPT kernel. The important idea here is that the runner is not "the model" itself. It is the harness around the model. In this repo, fixture generation lives only in Python; C++ uses those Python-generated fixtures for validation.

- `dump-fixtures` creates deterministic reference outputs
- `validate` checks another implementation against the reference
- `benchmark` repeatedly runs forward + backward on dataset examples

For the running example, the runner eventually turns `anna` into token IDs, builds a model with vocab size `27`, and sends that token sequence into the forward/backward kernel.

### Python

From `methods/serial_python/serial.py`:

```python
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
    ...


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
    ...


def run_benchmark(dataset_path: Path, seed: int, preset_name: str, num_steps: int | None):
    docs = load_docs(dataset_path)
    uchars, vocab, bos = build_vocab(docs)
    preset = BENCHMARK_PRESETS[preset_name]
    config = preset["config"]
    requested_steps = num_steps if num_steps is not None else preset["steps"]
    steps = min(requested_steps, len(docs))
    state_dict, params = init_model(config, len(uchars) + 1, seed)
    last_result = None

    for step_idx in range(steps):
        tokens = encode_doc(docs[step_idx], vocab, bos)
        last_result = run_forward_backward(tokens, state_dict, params, config)
    ...


def main():
    args = parse_args()
    if args.mode == "dump-fixtures":
        dump_fixtures(args.dataset, args.fixture_dir, args.sample_name, args.seed)
        return
    if args.mode == "validate":
        validate_fixture(args.fixture_dir, args.seed)
        return
    run_benchmark(args.dataset, args.seed, args.preset, args.num_steps)
```

### C++

From `methods/serial_cpp/main.cpp`:

```cpp
struct CliOptions {
    std::string mode;
    std::filesystem::path fixture_dir;
    std::filesystem::path dataset;
    std::string sample_name = "anna";
    std::string preset = "small";
    int num_steps = -1;
    std::uint32_t seed = 42;
};

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg == "--mode") {
            options.mode = require_value(argc, argv, &idx);
        } else if (arg == "--fixture-dir") {
            options.fixture_dir = require_value(argc, argv, &idx);
        } else if (arg == "--dataset") {
            options.dataset = require_value(argc, argv, &idx);
        } else if (arg == "--sample-name") {
            options.sample_name = require_value(argc, argv, &idx);
        } else if (arg == "--preset") {
            options.preset = require_value(argc, argv, &idx);
        } else if (arg == "--num-steps") {
            options.num_steps = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(require_value(argc, argv, &idx)));
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.mode.empty()) {
        throw std::runtime_error("--mode is required");
    }
    return options;
}

int run_validate(const CliOptions& options) {
    const auto manifest = parse_manifest(options.fixture_dir / "manifest.txt");
    ModelConfig config;
    config.n_layer = std::stoi(manifest.at("n_layer"));
    config.n_embd = std::stoi(manifest.at("n_embd"));
    config.block_size = std::stoi(manifest.at("block_size"));
    config.n_head = std::stoi(manifest.at("n_head"));
    config.vocab_size = std::stoi(manifest.at("vocab_size"));

    const std::vector<int> tokens = parse_int_list(manifest.at("token_ids"));
    Model model = make_empty_model(config);
    load_model_from_f32(model, read_f32_file(options.fixture_dir / manifest.at("weights_init_file")));
    const KernelResult result = run_forward_backward(model, tokens);
    ...
    return 0;
}

int run_benchmark(const CliOptions& options) {
    const auto preset_it = kBenchmarkPresets.find(options.preset);
    const std::vector<std::string> docs = load_docs(options.dataset);
    const auto [uchars, vocab] = build_vocab(docs);
    BenchmarkPreset preset = preset_it->second;
    preset.config.vocab_size = static_cast<int>(uchars.size()) + 1;
    const int requested_steps = options.num_steps >= 0 ? options.num_steps : preset.steps;
    const int steps = std::min(requested_steps, static_cast<int>(docs.size()));
    const Model model = initialize_model(preset.config, options.seed);
    double last_loss = 0.0;
    std::string last_doc;

    for (int step = 0; step < steps; ++step) {
        last_doc = docs[step];
        const std::vector<int> tokens = encode_doc(docs[step], vocab, static_cast<int>(uchars.size()));
        last_loss = run_forward_backward(model, tokens).loss;
    }
    ...
    return 0;
}

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.mode == "validate") {
            return run_validate(options);
        }
        if (options.mode == "benchmark") {
            return run_benchmark(options);
        }
        throw std::runtime_error("unsupported mode: " + options.mode);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
```

## 2. Turning Names into Tokens

Before the model can reason about a name, both runners convert raw text into integer IDs. Conceptually, this section defines the model's vocabulary and turns a document into a sequence the model can step through one position at a time. A special beginning-of-sequence token is added to both ends so the model always predicts the next character from a known boundary.

Using the verified dataset character set in this repo:

- `a` maps to `0`
- `n` maps to `13`
- `BOS` maps to `26`

So `anna` becomes:

```text
[26, 0, 13, 13, 0, 26]
```

That means the model sees five training positions:

- input `26`, target `0`
- input `0`, target `13`
- input `13`, target `13`
- input `13`, target `0`
- input `0`, target `26`

### Python

From `methods/serial_python/serial.py`:

```python
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
```

### C++

From `methods/serial_cpp/main.cpp`:

```cpp
std::vector<std::string> load_docs(const std::filesystem::path& dataset_path) {
    std::ifstream input(dataset_path);
    if (!input) {
        throw std::runtime_error("failed to open dataset: " + dataset_path.string());
    }
    std::vector<std::string> docs;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            docs.push_back(line);
        }
    }
    return docs;
}

std::pair<std::string, std::unordered_map<char, int>> build_vocab(const std::vector<std::string>& docs) {
    std::string chars;
    for (const std::string& doc : docs) {
        chars += doc;
    }
    std::sort(chars.begin(), chars.end());
    chars.erase(std::unique(chars.begin(), chars.end()), chars.end());

    std::unordered_map<char, int> vocab;
    for (int idx = 0; idx < static_cast<int>(chars.size()); ++idx) {
        vocab[chars[idx]] = idx;
    }
    return {chars, vocab};
}

std::vector<int> encode_doc(const std::string& doc, const std::unordered_map<char, int>& vocab, int bos_token_id) {
    std::vector<int> tokens;
    tokens.reserve(doc.size() + 2);
    tokens.push_back(bos_token_id);
    for (char ch : doc) {
        const auto it = vocab.find(ch);
        if (it == vocab.end()) {
            throw std::runtime_error(std::string("sample name contains character not in dataset: ") + ch);
        }
        tokens.push_back(it->second);
    }
    tokens.push_back(bos_token_id);
    return tokens;
}
```

## 3. Model Shape and Parameter Initialization

This section defines the size of the network and allocates its learned parameters. Conceptually, this is where the architecture is fixed: how many layers exist, how wide the hidden state is, how many attention heads there are, and how many weights must be learned. The Python version stores parameters as nested lists of `Value` objects. The C++ version stores them as flat `std::vector<double>` buffers.

For `anna`, the important part is `vocab_size = 27`. That gives the model:

- 27 token embedding vectors in `wte`
- `block_size` position embedding vectors in `wpe`
- an output head that scores 27 possible next tokens at every position

### Python

From `methods/serial_python/kernel.py`:

```python
@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 1
    n_embd: int = 16
    block_size: int = 16
    n_head: int = 4

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def state_dict_order(config):
    names = ["wte", "wpe", "lm_head"]
    for layer_idx in range(config.n_layer):
        names.extend(
            [
                f"layer{layer_idx}.attn_wq",
                f"layer{layer_idx}.attn_wk",
                f"layer{layer_idx}.attn_wv",
                f"layer{layer_idx}.attn_wo",
                f"layer{layer_idx}.mlp_fc1",
                f"layer{layer_idx}.mlp_fc2",
            ]
        )
    return names


def matrix(rng, nout, nin, std=0.08):
    return [[Value(rng.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def init_model(config, vocab_size, seed):
    rng = random.Random(seed)
    state_dict = {
        "wte": matrix(rng, vocab_size, config.n_embd),
        "wpe": matrix(rng, config.block_size, config.n_embd),
        "lm_head": matrix(rng, vocab_size, config.n_embd),
    }
    for layer_idx in range(config.n_layer):
        state_dict[f"layer{layer_idx}.attn_wq"] = matrix(rng, config.n_embd, config.n_embd)
        state_dict[f"layer{layer_idx}.attn_wk"] = matrix(rng, config.n_embd, config.n_embd)
        state_dict[f"layer{layer_idx}.attn_wv"] = matrix(rng, config.n_embd, config.n_embd)
        state_dict[f"layer{layer_idx}.attn_wo"] = matrix(rng, config.n_embd, config.n_embd)
        state_dict[f"layer{layer_idx}.mlp_fc1"] = matrix(rng, 4 * config.n_embd, config.n_embd)
        state_dict[f"layer{layer_idx}.mlp_fc2"] = matrix(rng, config.n_embd, 4 * config.n_embd)
    params = flatten_params(state_dict, config)
    return state_dict, params


def flatten_params(state_dict, config):
    params = []
    for name in state_dict_order(config):
        for row in state_dict[name]:
            params.extend(row)
    return params
```

### C++

From `methods/serial_cpp/kernel.hpp` and `methods/serial_cpp/kernel.cpp`:

```cpp
struct ModelConfig {
    int n_layer = 1;
    int n_embd = 16;
    int block_size = 16;
    int n_head = 4;
    int vocab_size = 0;

    int head_dim() const { return n_embd / n_head; }
    int mlp_dim() const { return 4 * n_embd; }
};

struct LayerWeights {
    std::vector<double> attn_wq;
    std::vector<double> attn_wk;
    std::vector<double> attn_wv;
    std::vector<double> attn_wo;
    std::vector<double> mlp_fc1;
    std::vector<double> mlp_fc2;
};

struct Model {
    ModelConfig config;
    std::vector<double> wte;
    std::vector<double> wpe;
    std::vector<double> lm_head;
    std::vector<LayerWeights> layers;
};
```

```cpp
Model make_empty_model(const ModelConfig& config) {
    Model model;
    model.config = config;
    model.wte.assign(config.vocab_size * config.n_embd, 0.0);
    model.wpe.assign(config.block_size * config.n_embd, 0.0);
    model.lm_head.assign(config.vocab_size * config.n_embd, 0.0);
    model.layers.resize(config.n_layer);
    for (LayerWeights& layer : model.layers) {
        layer.attn_wq.assign(config.n_embd * config.n_embd, 0.0);
        layer.attn_wk.assign(config.n_embd * config.n_embd, 0.0);
        layer.attn_wv.assign(config.n_embd * config.n_embd, 0.0);
        layer.attn_wo.assign(config.n_embd * config.n_embd, 0.0);
        layer.mlp_fc1.assign(config.mlp_dim() * config.n_embd, 0.0);
        layer.mlp_fc2.assign(config.n_embd * config.mlp_dim(), 0.0);
    }
    return model;
}

Model initialize_model(const ModelConfig& config, std::uint32_t seed) {
    Model model = make_empty_model(config);
    std::mt19937 rng(seed);
    assign_random(&model.wte, &rng);
    assign_random(&model.wpe, &rng);
    assign_random(&model.lm_head, &rng);
    for (LayerWeights& layer : model.layers) {
        assign_random(&layer.attn_wq, &rng);
        assign_random(&layer.attn_wk, &rng);
        assign_random(&layer.attn_wv, &rng);
        assign_random(&layer.attn_wo, &rng);
        assign_random(&layer.mlp_fc1, &rng);
        assign_random(&layer.mlp_fc2, &rng);
    }
    return model;
}
```

## 4. Math Engine and Gradient Strategy

Every GPT needs the same mathematical building blocks: matrix multiplies, normalization, softmax, nonlinearities, and gradients. The conceptual difference here is important:

- Python tracks every scalar as a `Value`, so the computation graph is built automatically during the forward pass and then traversed backward.
- C++ stores ordinary numbers during the forward pass and computes gradients manually with explicit backward routines and cached activations.

For `anna`, this means the model does not reason over the raw integers `26, 0, 13, 13, 0, 26` directly. Those IDs are only lookup keys. Once looked up, each token becomes a dense vector of length `n_embd`, and all later math happens on those vectors.

### Python

From `methods/serial_python/kernel.py`:

```python
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        exp_val = math.exp(self.data)
        return Value(exp_val, (self,), (exp_val,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            for child, local_grad in zip(node._children, node._local_grads):
                child.grad += local_grad * node.grad


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(value.data for value in logits)
    exps = [(value - max_val).exp() for value in logits]
    total = sum(exps)
    return [exp_value / total for exp_value in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

### C++

From `methods/serial_cpp/kernel.cpp`:

```cpp
std::vector<double> linear(const std::vector<double>& input, const std::vector<double>& weights, int out_dim) {
    const int in_dim = static_cast<int>(input.size());
    std::vector<double> output(out_dim, 0.0);
    for (int out = 0; out < out_dim; ++out) {
        double sum = 0.0;
        for (int in = 0; in < in_dim; ++in) {
            sum += weights[out * in_dim + in] * input[in];
        }
        output[out] = sum;
    }
    return output;
}

void backward_linear(
    const std::vector<double>& input,
    const std::vector<double>& weights,
    int out_dim,
    const std::vector<double>& doutput,
    std::vector<double>* dweights,
    std::vector<double>* dinput
) {
    const int in_dim = static_cast<int>(input.size());
    for (int out = 0; out < out_dim; ++out) {
        const double grad = doutput[out];
        if (grad == 0.0) {
            continue;
        }
        for (int in = 0; in < in_dim; ++in) {
            (*dweights)[out * in_dim + in] += grad * input[in];
            (*dinput)[in] += weights[out * in_dim + in] * grad;
        }
    }
}

std::vector<double> softmax(const std::vector<double>& logits) {
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size(), 0.0);
    double exp_sum = 0.0;
    for (std::size_t idx = 0; idx < logits.size(); ++idx) {
        probs[idx] = std::exp(logits[idx] - max_logit);
        exp_sum += probs[idx];
    }
    for (double& prob : probs) {
        prob /= exp_sum;
    }
    return probs;
}

std::vector<double> rmsnorm(const std::vector<double>& input) {
    double mean_square = 0.0;
    for (double value : input) {
        mean_square += value * value;
    }
    mean_square /= static_cast<double>(input.size());
    const double scale = 1.0 / std::sqrt(mean_square + kRmsNormEps);

    std::vector<double> output(input.size(), 0.0);
    for (std::size_t idx = 0; idx < input.size(); ++idx) {
        output[idx] = input[idx] * scale;
    }
    return output;
}

std::vector<double> backward_rmsnorm(const std::vector<double>& input, const std::vector<double>& doutput) {
    double mean_square = 0.0;
    double dot = 0.0;
    for (std::size_t idx = 0; idx < input.size(); ++idx) {
        mean_square += input[idx] * input[idx];
        dot += doutput[idx] * input[idx];
    }
    mean_square /= static_cast<double>(input.size());
    const double scale = 1.0 / std::sqrt(mean_square + kRmsNormEps);
    const double coeff = dot * scale * scale * scale / static_cast<double>(input.size());

    std::vector<double> dinput(input.size(), 0.0);
    for (std::size_t idx = 0; idx < input.size(); ++idx) {
        dinput[idx] = doutput[idx] * scale - input[idx] * coeff;
    }
    return dinput;
}
```

## 5. Input Representation: Token Embedding + Position Embedding

A transformer does not work directly on integer token IDs. It first converts the current token into a learned vector and adds a learned position vector so the model knows both what symbol it is seeing and where it sits in the sequence. This code constructs that starting representation before any attention is applied.

For `anna`, the first few embedding lookups are:

- position `0`: `x = wte[26] + wpe[0]` because the first token is `BOS`
- position `1`: `x = wte[0] + wpe[1]` because the second token is `a`
- position `2`: `x = wte[13] + wpe[2]` because the third token is `n`

So the token identity and the position identity are blended together before the transformer layers see anything.

### Python

From `methods/serial_python/kernel.py`:

```python
def gpt(token_id, pos_id, keys, values, state_dict, config):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [token_value + pos_value for token_value, pos_value in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    ...
```

### C++

From `methods/serial_cpp/kernel.cpp`:

```cpp
for (int pos = 0; pos < cache.seq_len; ++pos) {
    StepCache& step = cache.steps[pos];
    step.input_token = tokens[pos];
    step.target_token = tokens[pos + 1];
    step.embed_pre.assign(config.n_embd, 0.0);

    for (int col = 0; col < config.n_embd; ++col) {
        step.embed_pre[col] =
            model.wte[step.input_token * config.n_embd + col] +
            model.wpe[pos * config.n_embd + col];
    }

    std::vector<double> x = rmsnorm(step.embed_pre);
    step.x0 = x;
    ...
}
```

## 6. Transformer Block: Causal Self-Attention

This is the part that gives GPT its "context" behavior. For the current position, the model builds a query, key, and value. The query asks, "what information do I need right now?" The keys represent what past positions contain. The dot products produce attention scores, softmax turns those scores into weights, and the values from earlier positions are blended together using those weights.

Because each position only looks at positions up to itself, this is causal attention: the model can use the past but not the future.

With `anna`, that looks like this:

- at position `0`, the model only sees `BOS`
- at position `1`, it can attend to `BOS, a`
- at position `2`, it can attend to `BOS, a, n`
- at position `3`, it can attend to `BOS, a, n, n`
- at position `4`, it can attend to `BOS, a, n, n, a`

So when the model is trying to predict the token after the first `n`, it is allowed to look back at the earlier `BOS` and `a`, plus the current `n`, but not at any future token.

### Python

From `methods/serial_python/kernel.py`:

```python
for layer_idx in range(config.n_layer):
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, state_dict[f"layer{layer_idx}.attn_wq"])
    k = linear(x, state_dict[f"layer{layer_idx}.attn_wk"])
    v = linear(x, state_dict[f"layer{layer_idx}.attn_wv"])
    keys[layer_idx].append(k)
    values[layer_idx].append(v)
    x_attn = []

    for head_idx in range(config.n_head):
        hs = head_idx * config.head_dim
        q_h = q[hs : hs + config.head_dim]
        k_h = [key[hs : hs + config.head_dim] for key in keys[layer_idx]]
        v_h = [value[hs : hs + config.head_dim] for value in values[layer_idx]]
        attn_logits = [
            sum(q_h[j] * k_h[t][j] for j in range(config.head_dim)) / config.head_dim**0.5
            for t in range(len(k_h))
        ]
        attn_weights = softmax(attn_logits)
        head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(config.head_dim)]
        x_attn.extend(head_out)

    x = linear(x_attn, state_dict[f"layer{layer_idx}.attn_wo"])
    x = [left + right for left, right in zip(x, x_residual)]
```

### C++

From `methods/serial_cpp/kernel.cpp`:

```cpp
for (int layer_idx = 0; layer_idx < config.n_layer; ++layer_idx) {
    const LayerWeights& layer = model.layers[layer_idx];
    LayerStepCache& layer_step = step.layers[layer_idx];
    layer_step.x_in = x;
    layer_step.x_norm1 = rmsnorm(layer_step.x_in);
    layer_step.q = linear(layer_step.x_norm1, layer.attn_wq, config.n_embd);
    layer_step.k = linear(layer_step.x_norm1, layer.attn_wk, config.n_embd);
    layer_step.v = linear(layer_step.x_norm1, layer.attn_wv, config.n_embd);
    layer_step.x_attn.assign(config.n_embd, 0.0);
    layer_step.attn_weights.resize(config.n_head);

    for (int head = 0; head < config.n_head; ++head) {
        const int head_start = head * config.head_dim();
        std::vector<double> attn_logits(pos + 1, 0.0);
        for (int t = 0; t <= pos; ++t) {
            const std::vector<double>& past_k = cache.steps[t].layers[layer_idx].k;
            double dot = 0.0;
            for (int j = 0; j < config.head_dim(); ++j) {
                dot += layer_step.q[head_start + j] * past_k[head_start + j];
            }
            attn_logits[t] = dot / std::sqrt(static_cast<double>(config.head_dim()));
        }

        layer_step.attn_weights[head] = softmax(attn_logits);
        for (int t = 0; t <= pos; ++t) {
            const std::vector<double>& past_v = cache.steps[t].layers[layer_idx].v;
            const double weight = layer_step.attn_weights[head][t];
            for (int j = 0; j < config.head_dim(); ++j) {
                layer_step.x_attn[head_start + j] += weight * past_v[head_start + j];
            }
        }
    }

    const std::vector<double> attn_proj = linear(layer_step.x_attn, layer.attn_wo, config.n_embd);
    layer_step.x_mid = add_vectors(layer_step.x_in, attn_proj);
    ...
}
```

## 7. Transformer Block: Feed-Forward MLP

Attention mixes information across positions, but each position still needs a strong local transformation after that mixing. The feed-forward network expands the representation, applies a nonlinearity, compresses it back down, and adds another residual connection. Conceptually, this is where the model transforms the attended context into richer features for the next layer or the output head.

For `anna`, after attention has mixed in the useful earlier context, the MLP lets each position reinterpret that context locally. So the hidden state for the second `n` can be transformed differently from the hidden state for the first `n`, even though they started from the same character ID, because they sit at different positions and attended to different histories.

### Python

From `methods/serial_python/kernel.py`:

```python
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict[f"layer{layer_idx}.mlp_fc1"])
    x = [value.relu() for value in x]
    x = linear(x, state_dict[f"layer{layer_idx}.mlp_fc2"])
    x = [left + right for left, right in zip(x, x_residual)]
```

### C++

From `methods/serial_cpp/kernel.cpp`:

```cpp
    layer_step.x_mid = add_vectors(layer_step.x_in, attn_proj);
    layer_step.x_norm2 = rmsnorm(layer_step.x_mid);
    layer_step.fc1 = linear(layer_step.x_norm2, layer.mlp_fc1, config.mlp_dim());
    layer_step.relu = layer_step.fc1;
    for (double& value : layer_step.relu) {
        value = std::max(0.0, value);
    }
    const std::vector<double> fc2 = linear(layer_step.relu, layer.mlp_fc2, config.n_embd);
    x = add_vectors(layer_step.x_mid, fc2);
```

## 8. Output Head and Next-Token Loss

After the final transformer layer, the model turns the hidden state into vocabulary-sized scores called logits. Those logits are converted into probabilities with softmax, and the model is penalized based on how much probability it assigned to the real next token. This is the standard next-token prediction objective.

For `anna`, the model produces five logits vectors, each of length `27`:

- after seeing `BOS`, score the probability of the next token being `a`
- after seeing `a`, score the probability of the next token being `n`
- after seeing the first `n`, score the probability of the next token being `n`
- after seeing the second `n`, score the probability of the next token being `a`
- after seeing the final `a`, score the probability of the next token being `BOS`

### Python

From `methods/serial_python/kernel.py`:

```python
def gpt(token_id, pos_id, keys, values, state_dict, config):
    ...
    return linear(x, state_dict["lm_head"])


def run_forward_backward(tokens, state_dict, params, config):
    zero_grads(params)
    seq_len = min(config.block_size, len(tokens) - 1)
    keys = [[] for _ in range(config.n_layer)]
    values = [[] for _ in range(config.n_layer)]
    logits_per_position = []
    losses = []
    for pos_id in range(seq_len):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values, state_dict, config)
        logits_per_position.append(logits)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
    loss = (1 / seq_len) * sum(losses)
    ...
```

### C++

From `methods/serial_cpp/kernel.cpp`:

```cpp
step.final_x = x;
step.logits = linear(step.final_x, model.lm_head, config.vocab_size);
const std::vector<double> probs = softmax(step.logits);
cache.loss += -std::log(probs[step.target_token]);
...
cache.loss /= static_cast<double>(cache.seq_len);
```

## 9. Full Sequence Pass and Backward Pass

The model does not process the whole sequence in one giant matrix operation here. Instead, it walks left-to-right through the token sequence. At each position it produces logits for the next token, accumulates loss, and keeps the keys and values needed so later positions can attend to earlier ones.

Then the backward pass pushes the loss signal in reverse:

- from logits into the final hidden state
- backward through the MLP
- backward through attention
- backward into embeddings and positional encodings

This is the complete training step for this project, except for the optimizer update.

For `anna`, the full pass is:

1. encode `anna` as `[26, 0, 13, 13, 0, 26]`
2. run position `0` to predict `a`
3. run position `1` to predict the first `n`
4. run position `2` to predict the second `n`
5. run position `3` to predict `a`
6. run position `4` to predict `BOS`
7. average those five losses
8. backpropagate gradients into all weights, including `wte` and `wpe`

### Python

From `methods/serial_python/kernel.py`:

```python
def run_forward_backward(tokens, state_dict, params, config):
    zero_grads(params)
    seq_len = min(config.block_size, len(tokens) - 1)
    keys = [[] for _ in range(config.n_layer)]
    values = [[] for _ in range(config.n_layer)]
    logits_per_position = []
    losses = []

    for pos_id in range(seq_len):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values, state_dict, config)
        logits_per_position.append(logits)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())

    loss = (1 / seq_len) * sum(losses)
    loss.backward()
    return {
        "seq_len": seq_len,
        "logits": [[value.data for value in row] for row in logits_per_position],
        "loss": loss.data,
        "grads": flatten_param_grads(params),
    }
```

### C++

From `methods/serial_cpp/kernel.cpp`:

```cpp
KernelResult run_forward_backward(const Model& model, const std::vector<int>& tokens) {
    if (tokens.size() < 2) {
        throw std::runtime_error("token sequence must contain at least one input and one target token");
    }

    const ForwardCache cache = forward_pass(model, tokens);
    KernelResult result;
    result.seq_len = cache.seq_len;
    result.loss = cache.loss;
    result.grads = backward_pass(model, cache);

    for (const StepCache& step : cache.steps) {
        append_values(&result.logits, step.logits);
    }
    return result;
}
```

```cpp
Model backward_pass(const Model& model, const ForwardCache& cache) {
    const ModelConfig& config = model.config;
    Model grads = make_empty_model(config);

    std::vector<std::vector<double>> dx(cache.seq_len, std::vector<double>(config.n_embd, 0.0));
    for (int pos = 0; pos < cache.seq_len; ++pos) {
        std::vector<double> dlogits = softmax(cache.steps[pos].logits);
        dlogits[cache.steps[pos].target_token] -= 1.0;
        for (double& grad : dlogits) {
            grad /= static_cast<double>(cache.seq_len);
        }
        backward_linear(cache.steps[pos].final_x, model.lm_head, config.vocab_size, dlogits, &grads.lm_head, &dx[pos]);
    }

    for (int layer_idx = config.n_layer - 1; layer_idx >= 0; --layer_idx) {
        std::vector<std::vector<double>> dx_prev(cache.seq_len, std::vector<double>(config.n_embd, 0.0));
        std::vector<std::vector<double>> d_q(cache.seq_len, std::vector<double>(config.n_embd, 0.0));
        std::vector<std::vector<double>> d_k(cache.seq_len, std::vector<double>(config.n_embd, 0.0));
        std::vector<std::vector<double>> d_v(cache.seq_len, std::vector<double>(config.n_embd, 0.0));
        LayerWeights& layer_grads = grads.layers[layer_idx];
        const LayerWeights& layer = model.layers[layer_idx];

        for (int pos = cache.seq_len - 1; pos >= 0; --pos) {
            const LayerStepCache& layer_step = cache.steps[pos].layers[layer_idx];
            std::vector<double> d_x_mid = dx[pos];

            std::vector<double> d_relu(config.mlp_dim(), 0.0);
            backward_linear(layer_step.relu, layer.mlp_fc2, config.n_embd, dx[pos], &layer_grads.mlp_fc2, &d_relu);

            std::vector<double> d_fc1(config.mlp_dim(), 0.0);
            for (int idx = 0; idx < config.mlp_dim(); ++idx) {
                d_fc1[idx] = layer_step.fc1[idx] > 0.0 ? d_relu[idx] : 0.0;
            }

            std::vector<double> d_x_norm2(config.n_embd, 0.0);
            backward_linear(layer_step.x_norm2, layer.mlp_fc1, config.mlp_dim(), d_fc1, &layer_grads.mlp_fc1, &d_x_norm2);
            add_in_place(&d_x_mid, backward_rmsnorm(layer_step.x_mid, d_x_norm2));

            std::vector<double> d_x_in = d_x_mid;
            std::vector<double> d_x_attn(config.n_embd, 0.0);
            backward_linear(layer_step.x_attn, layer.attn_wo, config.n_embd, d_x_mid, &layer_grads.attn_wo, &d_x_attn);

            for (int head = 0; head < config.n_head; ++head) {
                const int head_start = head * config.head_dim();
                const double scale = 1.0 / std::sqrt(static_cast<double>(config.head_dim()));
                std::vector<double> d_attn_weights(pos + 1, 0.0);

                for (int t = 0; t <= pos; ++t) {
                    const std::vector<double>& past_v = cache.steps[t].layers[layer_idx].v;
                    double dot = 0.0;
                    for (int j = 0; j < config.head_dim(); ++j) {
                        dot += d_x_attn[head_start + j] * past_v[head_start + j];
                        d_v[t][head_start + j] += layer_step.attn_weights[head][t] * d_x_attn[head_start + j];
                    }
                    d_attn_weights[t] = dot;
                }

                double softmax_dot = 0.0;
                for (int t = 0; t <= pos; ++t) {
                    softmax_dot += d_attn_weights[t] * layer_step.attn_weights[head][t];
                }

                for (int t = 0; t <= pos; ++t) {
                    const std::vector<double>& past_k = cache.steps[t].layers[layer_idx].k;
                    const double dlogit = layer_step.attn_weights[head][t] * (d_attn_weights[t] - softmax_dot);
                    for (int j = 0; j < config.head_dim(); ++j) {
                        d_q[pos][head_start + j] += dlogit * past_k[head_start + j] * scale;
                        d_k[t][head_start + j] += dlogit * layer_step.q[head_start + j] * scale;
                    }
                }
            }

            std::vector<double> d_x_norm1(config.n_embd, 0.0);
            backward_linear(layer_step.x_norm1, layer.attn_wq, config.n_embd, d_q[pos], &layer_grads.attn_wq, &d_x_norm1);
            backward_linear(layer_step.x_norm1, layer.attn_wk, config.n_embd, d_k[pos], &layer_grads.attn_wk, &d_x_norm1);
            backward_linear(layer_step.x_norm1, layer.attn_wv, config.n_embd, d_v[pos], &layer_grads.attn_wv, &d_x_norm1);
            add_in_place(&d_x_in, backward_rmsnorm(layer_step.x_in, d_x_norm1));

            dx_prev[pos] = std::move(d_x_in);
        }

        dx = std::move(dx_prev);
    }

    for (int pos = 0; pos < cache.seq_len; ++pos) {
        const std::vector<double> d_embed_pre = backward_rmsnorm(cache.steps[pos].embed_pre, dx[pos]);
        const int token_id = cache.steps[pos].input_token;
        for (int col = 0; col < config.n_embd; ++col) {
            const double grad = d_embed_pre[col];
            grads.wte[token_id * config.n_embd + col] += grad;
            grads.wpe[pos * config.n_embd + col] += grad;
        }
    }

    return grads;
}
```

## 10. Big Picture Summary

If you strip away the implementation details, both versions are doing the same conceptual job:

1. read a name and convert it into token IDs
2. map each token position to an embedding vector
3. repeatedly refine that vector with causal attention and an MLP
4. produce logits over the vocabulary for the next token
5. compute next-token loss
6. backpropagate gradients through the whole sequence

For the concrete example `anna`, the whole model is just learning this character-level next-token chain:

```text
BOS -> a -> n -> n -> a -> BOS
```

Everything else in the transformer exists to turn that simple sequence into useful hidden states and gradients.

The main difference is implementation style:

- Python is the clarity-first reference, using scalar autodiff.
- C++ is the explicit systems version, using flat buffers, caches, and handwritten backward logic.

That makes the Python code easier to read as a model description, while the C++ code is closer to how a performance-oriented implementation is usually organized.

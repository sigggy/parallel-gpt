import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 1
    n_embd: int = 16
    block_size: int = 16
    n_head: int = 4

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class Value:
    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data)

    def __pow__(self, other):
        return Value(self.data**other)

    def log(self):
        return Value(math.log(self.data))

    def exp(self):
        return Value(math.exp(self.data))

    def relu(self):
        return Value(max(0, self.data))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

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


def flatten_param_values(state_dict, config):
    return [param.data for param in flatten_params(state_dict, config)]


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


def gpt(token_id, pos_id, keys, values, state_dict, config):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [token_value + pos_value for token_value, pos_value in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

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
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{layer_idx}.mlp_fc1"])
        x = [value.relu() for value in x]
        x = linear(x, state_dict[f"layer{layer_idx}.mlp_fc2"])
        x = [left + right for left, right in zip(x, x_residual)]

    return linear(x, state_dict["lm_head"])


def run_forward(tokens, state_dict, config):
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
    return {
        "seq_len": seq_len,
        "logits": [[value.data for value in row] for row in logits_per_position],
        "loss": loss.data,
    }

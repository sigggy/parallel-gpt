#include "kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

struct LayerCache {
    std::vector<double> x_in;
    std::vector<double> x_norm1;
    std::vector<double> q;
    std::vector<double> k;
    std::vector<double> v;
    std::vector<double> attn_weights;
    std::vector<double> attn_concat;
    std::vector<double> x_mid;
    std::vector<double> x_norm2;
    std::vector<double> fc1;
    std::vector<double> relu;
};

struct ForwardCache {
    int seq_len = 0;
    std::vector<int> input_tokens;
    std::vector<int> target_tokens;
    std::vector<double> embed_pre;
    std::vector<double> x0;
    std::vector<LayerCache> layers;
    std::vector<double> final_x;
    std::vector<double> logits;
    double loss = 0.0;
};

constexpr double kRmsNormEps = 1e-5;
constexpr double kInitStd = 0.08;

void append_values(std::vector<double>* out, const std::vector<double>& values) {
    out->insert(out->end(), values.begin(), values.end());
}

std::vector<double> rmsnorm_rows(const std::vector<double>& input, int rows, int cols) {
    std::vector<double> output(input.size(), 0.0);
    for (int row = 0; row < rows; ++row) {
        double mean_square = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double value = input[row * cols + col];
            mean_square += value * value;
        }
        mean_square /= static_cast<double>(cols);
        const double scale = 1.0 / std::sqrt(mean_square + kRmsNormEps);
        for (int col = 0; col < cols; ++col) {
            output[row * cols + col] = input[row * cols + col] * scale;
        }
    }
    return output;
}

std::vector<double> backward_rmsnorm_rows(const std::vector<double>& input, const std::vector<double>& doutput, int rows, int cols) {
    std::vector<double> dinput(input.size(), 0.0);
    for (int row = 0; row < rows; ++row) {
        double mean_square = 0.0;
        double dot = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double value = input[row * cols + col];
            mean_square += value * value;
            dot += doutput[row * cols + col] * value;
        }
        mean_square /= static_cast<double>(cols);
        const double scale = 1.0 / std::sqrt(mean_square + kRmsNormEps);
        const double coeff = dot * scale * scale * scale / static_cast<double>(cols);
        for (int col = 0; col < cols; ++col) {
            const double value = input[row * cols + col];
            dinput[row * cols + col] = doutput[row * cols + col] * scale - value * coeff;
        }
    }
    return dinput;
}

std::vector<double> linear_rows(const std::vector<double>& input, int rows, int in_dim, const std::vector<double>& weights, int out_dim) {
    std::vector<double> output(rows * out_dim, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < out_dim; ++out) {
            double sum = 0.0;
            for (int in = 0; in < in_dim; ++in) {
                sum += weights[out * in_dim + in] * input[row * in_dim + in];
            }
            output[row * out_dim + out] = sum;
        }
    }
    return output;
}

void backward_linear_rows(
    const std::vector<double>& input,
    int rows,
    int in_dim,
    const std::vector<double>& weights,
    int out_dim,
    const std::vector<double>& doutput,
    std::vector<double>* dweights,
    std::vector<double>* dinput
) {
    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < out_dim; ++out) {
            const double grad = doutput[row * out_dim + out];
            if (grad == 0.0) {
                continue;
            }
            for (int in = 0; in < in_dim; ++in) {
                (*dweights)[out * in_dim + in] += grad * input[row * in_dim + in];
                (*dinput)[row * in_dim + in] += weights[out * in_dim + in] * grad;
            }
        }
    }
}

std::vector<double> add_vectors(const std::vector<double>& left, const std::vector<double>& right) {
    std::vector<double> output(left.size(), 0.0);
    for (std::size_t idx = 0; idx < left.size(); ++idx) {
        output[idx] = left[idx] + right[idx];
    }
    return output;
}

ForwardCache forward_pass(const Model& model, const std::vector<int>& tokens) {
    const ModelConfig& config = model.config;
    const int seq_len = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);
    const int embd = config.n_embd;
    const int vocab_size = config.vocab_size;
    ForwardCache cache;
    cache.seq_len = seq_len;
    cache.input_tokens.assign(tokens.begin(), tokens.begin() + seq_len);
    cache.target_tokens.assign(tokens.begin() + 1, tokens.begin() + 1 + seq_len);
    cache.embed_pre.assign(seq_len * embd, 0.0);

    for (int pos = 0; pos < seq_len; ++pos) {
        const int token_id = cache.input_tokens[pos];
        for (int col = 0; col < embd; ++col) {
            cache.embed_pre[pos * embd + col] = model.wte[token_id * embd + col] + model.wpe[pos * embd + col];
        }
    }

    std::vector<double> x = rmsnorm_rows(cache.embed_pre, seq_len, embd);
    cache.x0 = x;
    cache.layers.resize(config.n_layer);

    for (int layer_idx = 0; layer_idx < config.n_layer; ++layer_idx) {
        const LayerWeights& layer = model.layers[layer_idx];
        LayerCache& layer_cache = cache.layers[layer_idx];
        layer_cache.x_in = x;
        layer_cache.x_norm1 = rmsnorm_rows(layer_cache.x_in, seq_len, embd);
        layer_cache.q = linear_rows(layer_cache.x_norm1, seq_len, embd, layer.attn_wq, embd);
        layer_cache.k = linear_rows(layer_cache.x_norm1, seq_len, embd, layer.attn_wk, embd);
        layer_cache.v = linear_rows(layer_cache.x_norm1, seq_len, embd, layer.attn_wv, embd);
        layer_cache.attn_weights.assign(config.n_head * seq_len * seq_len, 0.0);
        layer_cache.attn_concat.assign(seq_len * embd, 0.0);

        for (int head = 0; head < config.n_head; ++head) {
            const int head_start = head * config.head_dim();
            for (int pos = 0; pos < seq_len; ++pos) {
                std::vector<double> attn_logits(pos + 1, 0.0);
                double max_logit = -std::numeric_limits<double>::infinity();
                for (int t = 0; t <= pos; ++t) {
                    double dot = 0.0;
                    for (int j = 0; j < config.head_dim(); ++j) {
                        dot += layer_cache.q[pos * embd + head_start + j] * layer_cache.k[t * embd + head_start + j];
                    }
                    attn_logits[t] = dot / std::sqrt(static_cast<double>(config.head_dim()));
                    max_logit = std::max(max_logit, attn_logits[t]);
                }
                std::vector<double> exps(pos + 1, 0.0);
                double exp_sum = 0.0;
                for (int t = 0; t <= pos; ++t) {
                    exps[t] = std::exp(attn_logits[t] - max_logit);
                    exp_sum += exps[t];
                }
                for (int t = 0; t <= pos; ++t) {
                    const double weight = exps[t] / exp_sum;
                    layer_cache.attn_weights[head * seq_len * seq_len + pos * seq_len + t] = weight;
                    for (int j = 0; j < config.head_dim(); ++j) {
                        layer_cache.attn_concat[pos * embd + head_start + j] += weight * layer_cache.v[t * embd + head_start + j];
                    }
                }
            }
        }

        const std::vector<double> attn_proj = linear_rows(layer_cache.attn_concat, seq_len, embd, layer.attn_wo, embd);
        layer_cache.x_mid = add_vectors(layer_cache.x_in, attn_proj);
        layer_cache.x_norm2 = rmsnorm_rows(layer_cache.x_mid, seq_len, embd);
        layer_cache.fc1 = linear_rows(layer_cache.x_norm2, seq_len, embd, layer.mlp_fc1, config.mlp_dim());
        layer_cache.relu = layer_cache.fc1;
        for (double& value : layer_cache.relu) {
            value = std::max(0.0, value);
        }
        const std::vector<double> fc2 = linear_rows(layer_cache.relu, seq_len, config.mlp_dim(), layer.mlp_fc2, embd);
        x = add_vectors(layer_cache.x_mid, fc2);
    }

    cache.final_x = x;
    cache.logits = linear_rows(cache.final_x, seq_len, embd, model.lm_head, vocab_size);
    cache.loss = 0.0;
    for (int pos = 0; pos < seq_len; ++pos) {
        double max_logit = -std::numeric_limits<double>::infinity();
        for (int vocab = 0; vocab < vocab_size; ++vocab) {
            max_logit = std::max(max_logit, cache.logits[pos * vocab_size + vocab]);
        }
        double exp_sum = 0.0;
        for (int vocab = 0; vocab < vocab_size; ++vocab) {
            exp_sum += std::exp(cache.logits[pos * vocab_size + vocab] - max_logit);
        }
        const int target = cache.target_tokens[pos];
        const double target_prob = std::exp(cache.logits[pos * vocab_size + target] - max_logit) / exp_sum;
        cache.loss += -std::log(target_prob);
    }
    cache.loss /= static_cast<double>(seq_len);
    return cache;
}

Model backward_pass(const Model& model, const ForwardCache& cache) {
    const ModelConfig& config = model.config;
    const int seq_len = cache.seq_len;
    const int embd = config.n_embd;
    const int vocab_size = config.vocab_size;
    Model grads = make_empty_model(config);

    std::vector<double> dlogits(seq_len * vocab_size, 0.0);
    for (int pos = 0; pos < seq_len; ++pos) {
        double max_logit = -std::numeric_limits<double>::infinity();
        for (int vocab = 0; vocab < vocab_size; ++vocab) {
            max_logit = std::max(max_logit, cache.logits[pos * vocab_size + vocab]);
        }
        std::vector<double> probs(vocab_size, 0.0);
        double exp_sum = 0.0;
        for (int vocab = 0; vocab < vocab_size; ++vocab) {
            probs[vocab] = std::exp(cache.logits[pos * vocab_size + vocab] - max_logit);
            exp_sum += probs[vocab];
        }
        for (int vocab = 0; vocab < vocab_size; ++vocab) {
            probs[vocab] /= exp_sum;
            dlogits[pos * vocab_size + vocab] = probs[vocab] / static_cast<double>(seq_len);
        }
        dlogits[pos * vocab_size + cache.target_tokens[pos]] -= 1.0 / static_cast<double>(seq_len);
    }

    std::vector<double> dx(cache.final_x.size(), 0.0);
    backward_linear_rows(cache.final_x, seq_len, embd, model.lm_head, vocab_size, dlogits, &grads.lm_head, &dx);

    for (int layer_idx = config.n_layer - 1; layer_idx >= 0; --layer_idx) {
        const LayerWeights& layer = model.layers[layer_idx];
        const LayerCache& layer_cache = cache.layers[layer_idx];
        LayerWeights& layer_grads = grads.layers[layer_idx];

        std::vector<double> d_fc2 = dx;
        std::vector<double> d_x_mid = dx;

        std::vector<double> d_relu(seq_len * config.mlp_dim(), 0.0);
        std::vector<double> d_x_norm2(seq_len * embd, 0.0);
        backward_linear_rows(layer_cache.relu, seq_len, config.mlp_dim(), layer.mlp_fc2, embd, d_fc2, &layer_grads.mlp_fc2, &d_relu);

        std::vector<double> d_fc1(seq_len * config.mlp_dim(), 0.0);
        for (std::size_t idx = 0; idx < d_fc1.size(); ++idx) {
            d_fc1[idx] = layer_cache.fc1[idx] > 0.0 ? d_relu[idx] : 0.0;
        }
        backward_linear_rows(layer_cache.x_norm2, seq_len, embd, layer.mlp_fc1, config.mlp_dim(), d_fc1, &layer_grads.mlp_fc1, &d_x_norm2);

        const std::vector<double> d_from_norm2 = backward_rmsnorm_rows(layer_cache.x_mid, d_x_norm2, seq_len, embd);
        for (std::size_t idx = 0; idx < d_x_mid.size(); ++idx) {
            d_x_mid[idx] += d_from_norm2[idx];
        }

        std::vector<double> d_attn_proj = d_x_mid;
        std::vector<double> d_x_in = d_x_mid;
        std::vector<double> d_attn_concat(seq_len * embd, 0.0);
        backward_linear_rows(layer_cache.attn_concat, seq_len, embd, layer.attn_wo, embd, d_attn_proj, &layer_grads.attn_wo, &d_attn_concat);

        std::vector<double> d_q(seq_len * embd, 0.0);
        std::vector<double> d_k(seq_len * embd, 0.0);
        std::vector<double> d_v(seq_len * embd, 0.0);

        for (int head = 0; head < config.n_head; ++head) {
            const int head_start = head * config.head_dim();
            const double scale = 1.0 / std::sqrt(static_cast<double>(config.head_dim()));
            std::vector<double> d_attn_weights(seq_len * seq_len, 0.0);

            for (int pos = 0; pos < seq_len; ++pos) {
                for (int t = 0; t <= pos; ++t) {
                    double dot = 0.0;
                    for (int j = 0; j < config.head_dim(); ++j) {
                        dot += d_attn_concat[pos * embd + head_start + j] * layer_cache.v[t * embd + head_start + j];
                        d_v[t * embd + head_start + j] += layer_cache.attn_weights[head * seq_len * seq_len + pos * seq_len + t] *
                                                          d_attn_concat[pos * embd + head_start + j];
                    }
                    d_attn_weights[pos * seq_len + t] += dot;
                }
            }

            for (int pos = 0; pos < seq_len; ++pos) {
                double softmax_dot = 0.0;
                for (int t = 0; t <= pos; ++t) {
                    const double weight = layer_cache.attn_weights[head * seq_len * seq_len + pos * seq_len + t];
                    softmax_dot += d_attn_weights[pos * seq_len + t] * weight;
                }
                for (int t = 0; t <= pos; ++t) {
                    const double weight = layer_cache.attn_weights[head * seq_len * seq_len + pos * seq_len + t];
                    const double dlogit = weight * (d_attn_weights[pos * seq_len + t] - softmax_dot);
                    for (int j = 0; j < config.head_dim(); ++j) {
                        d_q[pos * embd + head_start + j] += dlogit * layer_cache.k[t * embd + head_start + j] * scale;
                        d_k[t * embd + head_start + j] += dlogit * layer_cache.q[pos * embd + head_start + j] * scale;
                    }
                }
            }
        }

        std::vector<double> d_x_norm1(seq_len * embd, 0.0);
        backward_linear_rows(layer_cache.x_norm1, seq_len, embd, layer.attn_wq, embd, d_q, &layer_grads.attn_wq, &d_x_norm1);
        backward_linear_rows(layer_cache.x_norm1, seq_len, embd, layer.attn_wk, embd, d_k, &layer_grads.attn_wk, &d_x_norm1);
        backward_linear_rows(layer_cache.x_norm1, seq_len, embd, layer.attn_wv, embd, d_v, &layer_grads.attn_wv, &d_x_norm1);

        const std::vector<double> d_from_norm1 = backward_rmsnorm_rows(layer_cache.x_in, d_x_norm1, seq_len, embd);
        for (std::size_t idx = 0; idx < d_x_in.size(); ++idx) {
            d_x_in[idx] += d_from_norm1[idx];
        }
        dx = d_x_in;
    }

    const std::vector<double> d_embed_pre = backward_rmsnorm_rows(cache.embed_pre, dx, seq_len, embd);
    for (int pos = 0; pos < seq_len; ++pos) {
        const int token_id = cache.input_tokens[pos];
        for (int col = 0; col < embd; ++col) {
            const double grad = d_embed_pre[pos * embd + col];
            grads.wte[token_id * embd + col] += grad;
            grads.wpe[pos * embd + col] += grad;
        }
    }

    return grads;
}

void assign_random(std::vector<double>* values, std::mt19937* rng) {
    std::normal_distribution<double> normal(0.0, kInitStd);
    for (double& value : *values) {
        value = normal(*rng);
    }
}

}  // namespace

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

void load_model_from_f32(Model& model, const std::vector<float>& values) {
    std::size_t cursor = 0;
    auto load_into = [&](std::vector<double>* target) {
        if (cursor + target->size() > values.size()) {
            throw std::runtime_error("weights file is smaller than expected");
        }
        for (double& value : *target) {
            value = static_cast<double>(values[cursor++]);
        }
    };

    load_into(&model.wte);
    load_into(&model.wpe);
    load_into(&model.lm_head);
    for (LayerWeights& layer : model.layers) {
        load_into(&layer.attn_wq);
        load_into(&layer.attn_wk);
        load_into(&layer.attn_wv);
        load_into(&layer.attn_wo);
        load_into(&layer.mlp_fc1);
        load_into(&layer.mlp_fc2);
    }
    if (cursor != values.size()) {
        throw std::runtime_error("weights file is larger than expected");
    }
}

std::vector<double> flatten_model_values(const Model& model) {
    std::vector<double> values;
    std::size_t total_size = model.wte.size() + model.wpe.size() + model.lm_head.size();
    for (const LayerWeights& layer : model.layers) {
        total_size += layer.attn_wq.size() + layer.attn_wk.size() + layer.attn_wv.size() + layer.attn_wo.size() +
                      layer.mlp_fc1.size() + layer.mlp_fc2.size();
    }
    values.reserve(total_size);
    append_values(&values, model.wte);
    append_values(&values, model.wpe);
    append_values(&values, model.lm_head);
    for (const LayerWeights& layer : model.layers) {
        append_values(&values, layer.attn_wq);
        append_values(&values, layer.attn_wk);
        append_values(&values, layer.attn_wv);
        append_values(&values, layer.attn_wo);
        append_values(&values, layer.mlp_fc1);
        append_values(&values, layer.mlp_fc2);
    }
    return values;
}

KernelResult run_forward_backward(const Model& model, const std::vector<int>& tokens) {
    if (tokens.size() < 2) {
        throw std::runtime_error("token sequence must contain at least one input and one target token");
    }
    const ForwardCache cache = forward_pass(model, tokens);
    KernelResult result;
    result.seq_len = cache.seq_len;
    result.logits = cache.logits;
    result.loss = cache.loss;
    result.grads = backward_pass(model, cache);
    return result;
}

#include "kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

struct LayerStepCache {
    std::vector<double> x_in;
    std::vector<double> x_norm1;
    std::vector<double> q;
    std::vector<double> k;
    std::vector<double> v;
    std::vector<double> x_attn;
    std::vector<std::vector<double>> attn_weights;
    std::vector<double> x_mid;
    std::vector<double> x_norm2;
    std::vector<double> fc1;
    std::vector<double> relu;
};

struct StepCache {
    int input_token = 0;
    int target_token = 0;
    std::vector<double> embed_pre;
    std::vector<double> x0;
    std::vector<LayerStepCache> layers;
    std::vector<double> final_x;
    std::vector<double> logits;
};

struct ForwardCache {
    int seq_len = 0;
    std::vector<StepCache> steps;
    double loss = 0.0;
};

constexpr double kRmsNormEps = 1e-5;
constexpr double kInitStd = 0.08;

void append_values(std::vector<double>* out, const std::vector<double>& values) {
    out->insert(out->end(), values.begin(), values.end());
}

void add_in_place(std::vector<double>* target, const std::vector<double>& values) {
    for (std::size_t idx = 0; idx < target->size(); ++idx) {
        (*target)[idx] += values[idx];
    }
}

std::vector<double> add_vectors(const std::vector<double>& left, const std::vector<double>& right) {
    std::vector<double> output(left.size(), 0.0);
    for (std::size_t idx = 0; idx < left.size(); ++idx) {
        output[idx] = left[idx] + right[idx];
    }
    return output;
}

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

ForwardCache forward_pass(const Model& model, const std::vector<int>& tokens) {
    const ModelConfig& config = model.config;
    ForwardCache cache;
    cache.seq_len = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);
    cache.steps.resize(cache.seq_len);

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
        step.layers.resize(config.n_layer);

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
            layer_step.x_norm2 = rmsnorm(layer_step.x_mid);
            layer_step.fc1 = linear(layer_step.x_norm2, layer.mlp_fc1, config.mlp_dim());
            layer_step.relu = layer_step.fc1;
            for (double& value : layer_step.relu) {
                value = std::max(0.0, value);
            }
            const std::vector<double> fc2 = linear(layer_step.relu, layer.mlp_fc2, config.n_embd);
            x = add_vectors(layer_step.x_mid, fc2);
        }

        step.final_x = x;
        step.logits = linear(step.final_x, model.lm_head, config.vocab_size);
        const std::vector<double> probs = softmax(step.logits);
        cache.loss += -std::log(probs[step.target_token]);
    }

    cache.loss /= static_cast<double>(cache.seq_len);
    return cache;
}

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
    result.loss = cache.loss;
    result.grads = backward_pass(model, cache);

    for (const StepCache& step : cache.steps) {
        append_values(&result.logits, step.logits);
    }
    return result;
}

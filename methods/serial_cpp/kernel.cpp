#include "kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

void add_in_place(std::vector<double>* target, const std::vector<double>& values) {
    // Elementwise accumulation used for residual paths and gradient accumulation:
    // target[i] = target[i] + values[i]
    for (std::size_t idx = 0; idx < target->size(); ++idx) {
        (*target)[idx] += values[idx];
    }
}

std::vector<double> add_vectors(const std::vector<double>& left, const std::vector<double>& right) {
    // Pure elementwise vector add:
    // output[i] = left[i] + right[i]
    std::vector<double> output(left.size(), 0.0);
    for (std::size_t idx = 0; idx < left.size(); ++idx) {
        output[idx] = left[idx] + right[idx];
    }
    return output;
}

std::vector<double> linear(const std::vector<double>& input, const std::vector<double>& weights, int out_dim) {
    // Dense matrix-vector multiply.
    // Shapes:
    // - input: [in_dim]
    // - weights: [out_dim, in_dim] stored row-major
    // - output: [out_dim]
    //
    // Math:
    // output[out] = sum_in weights[out, in] * input[in]
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
    // Backward pass for the dense layer above.
    //
    // If y = W x, then:
    // - dW[out, in] += dY[out] * x[in]
    // - dX[in] += W[out, in] * dY[out]
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
    // Convert arbitrary logits into a probability distribution.
    //
    // Math:
    // probs[i] = exp(logits[i]) / sum_j exp(logits[j])
    //
    // We subtract max_logit first for numerical stability so exp() does not overflow.
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
    // RMSNorm rescales the vector by its root-mean-square magnitude.
    //
    // Math:
    // mean_square = (1 / N) * sum_i input[i]^2
    // scale = 1 / sqrt(mean_square + eps)
    // output[i] = input[i] * scale
    //
    // This keeps the direction of the vector but normalizes its overall scale.
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
    // Backward pass for RMSNorm.
    //
    // Intuition:
    // - one term scales the upstream gradient directly
    // - one correction term accounts for the fact that the normalization scale
    //   itself depends on every element of the input vector
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
    // Input: model weights plus one tokenized sequence such as [BOS, a, n, n, a, BOS].
    // Transformation: run embeddings, transformer blocks, and next-token scoring while caching activations.
    // Output: per-position hidden states/logits and the average next-token loss.
    const ModelConfig& config = model.config;
    ForwardCache cache;
    cache.seq_len = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);
    cache.steps.resize(cache.seq_len);

    //* Per token 
    for (int pos = 0; pos < cache.seq_len; ++pos) {
        StepCache& step = cache.steps[pos];
        step.input_token = tokens[pos];
        step.target_token = tokens[pos + 1];
        step.embed_pre.assign(config.n_embd, 0.0);

        // Input: one token ID and one position ID.
        // Transformation: look up the token embedding and position embedding, add them elementwise.
        // Output: the pre-normalized representation for this sequence position.
        //* CUDA speedup opporunity //* Per embedding dim
        for (int col = 0; col < config.n_embd; ++col) {
            double token_val = model.wte[step.input_token * config.n_embd + col];
            double pos_val   = model.wpe[pos * config.n_embd + col];

            step.embed_pre[col] = token_val + pos_val;
        }

        std::vector<double> x = rmsnorm(step.embed_pre); //* CUDA opporunity 
        step.x0 = x; // our first normalized hidden state 
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

            // Input: the current hidden state plus cached keys/values from earlier positions.
            // Transformation: build Q/K/V, compute causal attention weights, and mix prior value vectors.
            // Output: an attention-informed hidden vector for this layer.
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

            // Input: the post-attention hidden state.
            // Transformation: expand with FC1, apply ReLU, project back down with FC2, then add the residual.
            // Output: the layer output that becomes the next layer's input.
            const std::vector<double> fc2 = linear(layer_step.relu, layer.mlp_fc2, config.n_embd);
            x = add_vectors(layer_step.x_mid, fc2);
        }

        step.final_x = x;
        // Input: the final hidden state at this position.
        // Transformation: project into vocabulary space and normalize with softmax for the target token.
        // Output: logits for every possible next token and one scalar loss contribution.
        step.logits = linear(step.final_x, model.lm_head, config.vocab_size);
        const std::vector<double> probs = softmax(step.logits);
        cache.loss += -std::log(probs[step.target_token]);
    }

    cache.loss /= static_cast<double>(cache.seq_len);
    return cache;
}

Model backward_pass(const Model& model, const ForwardCache& cache) {
    // Input: the original model weights plus the cached forward activations.
    // Transformation: move the loss gradient backward through logits, MLP, attention, and embeddings.
    // Output: a gradient buffer with the same shape as the model weights.
    const ModelConfig& config = model.config;
    Model grads = make_empty_model(config);

    std::vector<std::vector<double>> dx(cache.seq_len, std::vector<double>(config.n_embd, 0.0));
    for (int pos = 0; pos < cache.seq_len; ++pos) {
        // Input: logits and the correct next-token ID.
        // Transformation: convert cross-entropy into gradients in vocabulary space.
        // Output: dlogits and the first hidden-state gradient for this position.
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

            // Input: gradient from the layer output.
            // Transformation: backprop through the MLP path and add the residual-path gradient.
            // Output: gradients for FC1/FC2 plus the gradient entering the attention sublayer.
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

            // Input: gradient on the attention output.
            // Transformation: distribute it back through attention weights and the stored Q/K/V paths.
            // Output: gradients for the attention projections and earlier timesteps' cached states.
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
        // Input: the gradient on the normalized embedding vector.
        // Transformation: undo the initial RMSNorm and split the result between token and position tables.
        // Output: updates for wte[token_id] and wpe[pos].
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

}  // namespace

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
        result.logits.insert(result.logits.end(), step.logits.begin(), step.logits.end());
    }
    return result;
}

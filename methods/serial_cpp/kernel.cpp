#include "kernel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

struct LayerStepCache {
    std::vector<double> k;
    std::vector<double> v;
};

struct StepForwardCache {
    std::vector<LayerStepCache> layers;
};

constexpr double kRmsNormEps = 1e-5;

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

KernelResult forward_pass(const Model& model, const std::vector<int>& tokens) {
    // Input: model weights plus one tokenized sequence such as [BOS, a, n, n, a, BOS].
    // Transformation: run embeddings, transformer blocks, and next-token scoring while caching activations.
    // Output: per-position hidden states/logits and the average next-token loss.
    const ModelConfig& config = model.config;
    KernelResult result;
    result.seq_len = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);
    std::vector<StepForwardCache> cache(result.seq_len);
    result.logits.reserve(static_cast<std::size_t>(result.seq_len) * static_cast<std::size_t>(config.vocab_size));

    //* Per token 
    for (int pos = 0; pos < result.seq_len; ++pos) {
        StepForwardCache& step_cache = cache[pos];
        const int input_token = tokens[pos];
        const int target_token = tokens[pos + 1];
        std::vector<double> embed_pre(config.n_embd, 0.0);

        // Input: one token ID and one position ID.
        // Transformation: look up the token embedding and position embedding, add them elementwise.
        // Output: the pre-normalized representation for this sequence position.
        //* CUDA speedup opporunity //* Per embedding dim
        for (int col = 0; col < config.n_embd; ++col) {
            double token_val = model.wte[input_token * config.n_embd + col];
            double pos_val   = model.wpe[pos * config.n_embd + col];

            embed_pre[col] = token_val + pos_val;
        }

        std::vector<double> x = rmsnorm(embed_pre); //* CUDA opporunity
        step_cache.layers.resize(config.n_layer);

        for (int layer_idx = 0; layer_idx < config.n_layer; ++layer_idx) {
            const LayerWeights& layer = model.layers[layer_idx];
            LayerStepCache& layer_step = step_cache.layers[layer_idx];
            const std::vector<double> x_residual = x;
            const std::vector<double> x_norm1 = rmsnorm(x);
            const std::vector<double> q = linear(x_norm1, layer.attn_wq, config.n_embd);
            layer_step.k = linear(x_norm1, layer.attn_wk, config.n_embd);
            layer_step.v = linear(x_norm1, layer.attn_wv, config.n_embd);
            std::vector<double> x_attn(config.n_embd, 0.0);

            // Input: the current hidden state plus cached keys/values from earlier positions.
            // Transformation: build Q/K/V, compute causal attention weights, and mix prior value vectors.
            // Output: an attention-informed hidden vector for this layer.
            for (int head = 0; head < config.n_head; ++head) {
                const int head_start = head * config.head_dim();
                std::vector<double> attn_logits(pos + 1, 0.0);
                for (int t = 0; t <= pos; ++t) {
                    const std::vector<double>& past_k = cache[t].layers[layer_idx].k;
                    double dot = 0.0;
                    for (int j = 0; j < config.head_dim(); ++j) {
                        dot += q[head_start + j] * past_k[head_start + j];
                    }
                    attn_logits[t] = dot / std::sqrt(static_cast<double>(config.head_dim()));
                }

                const std::vector<double> attn_weights = softmax(attn_logits);
                for (int t = 0; t <= pos; ++t) {
                    const std::vector<double>& past_v = cache[t].layers[layer_idx].v;
                    const double weight = attn_weights[t];
                    for (int j = 0; j < config.head_dim(); ++j) {
                        x_attn[head_start + j] += weight * past_v[head_start + j];
                    }
                }
            }

            const std::vector<double> attn_proj = linear(x_attn, layer.attn_wo, config.n_embd);
            const std::vector<double> x_mid = add_vectors(x_residual, attn_proj);
            const std::vector<double> x_norm2 = rmsnorm(x_mid);
            std::vector<double> relu = linear(x_norm2, layer.mlp_fc1, config.mlp_dim());
            for (double& value : relu) {
                value = std::max(0.0, value);
            }

            // Input: the post-attention hidden state.
            // Transformation: expand with FC1, apply ReLU, project back down with FC2, then add the residual.
            // Output: the layer output that becomes the next layer's input.
            const std::vector<double> fc2 = linear(relu, layer.mlp_fc2, config.n_embd);
            x = add_vectors(x_mid, fc2);
        }

        // Input: the final hidden state at this position.
        // Transformation: project into vocabulary space and normalize with softmax for the target token.
        // Output: logits for every possible next token and one scalar loss contribution.
        const std::vector<double> logits = linear(x, model.lm_head, config.vocab_size);
        const std::vector<double> probs = softmax(logits);
        result.logits.insert(result.logits.end(), logits.begin(), logits.end());
        result.loss += -std::log(probs[target_token]);
    }

    result.loss /= static_cast<double>(result.seq_len);
    return result;
}

}  // namespace

KernelResult run_forward(const Model& model, const std::vector<int>& tokens) {
    if (tokens.size() < 2) {
        throw std::runtime_error("token sequence must contain at least one input and one target token");
    }

    return forward_pass(model, tokens);
}

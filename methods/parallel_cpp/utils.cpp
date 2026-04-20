#include "kernel.hpp"

#include <random>
#include <stdexcept>

namespace {

constexpr double kInitStd = 0.08;

void append_values(std::vector<double>* out, const std::vector<double>& values) {
    out->insert(out->end(), values.begin(), values.end());
}

void assign_random(std::vector<double>* values, std::mt19937* rng) {
    std::normal_distribution<double> normal(0.0, kInitStd);
    for (double& value : *values) {
        value = normal(*rng);
    }
}

}  // namespace

Model make_empty_model(const ModelConfig& config) {
    Model host_model;
    host_model.config = config;
    host_model.wte.assign(config.vocab_size * config.n_embd, 0.0);
    host_model.wpe.assign(config.block_size * config.n_embd, 0.0);
    host_model.lm_head.assign(config.vocab_size * config.n_embd, 0.0);
    host_model.layers.resize(config.n_layer);
    for (LayerWeights& layer : host_model.layers) {
        layer.attn_wq.assign(config.n_embd * config.n_embd, 0.0);
        layer.attn_wk.assign(config.n_embd * config.n_embd, 0.0);
        layer.attn_wv.assign(config.n_embd * config.n_embd, 0.0);
        layer.attn_wo.assign(config.n_embd * config.n_embd, 0.0);
        layer.mlp_fc1.assign(config.mlp_dim() * config.n_embd, 0.0);
        layer.mlp_fc2.assign(config.n_embd * config.mlp_dim(), 0.0);
    }
    return host_model;
}

Model initialize_model(const ModelConfig& config, std::uint32_t seed) {
    Model host_model = make_empty_model(config);
    std::mt19937 rng(seed);
    assign_random(&host_model.wte, &rng);
    assign_random(&host_model.wpe, &rng);
    assign_random(&host_model.lm_head, &rng);
    for (LayerWeights& layer : host_model.layers) {
        assign_random(&layer.attn_wq, &rng);
        assign_random(&layer.attn_wk, &rng);
        assign_random(&layer.attn_wv, &rng);
        assign_random(&layer.attn_wo, &rng);
        assign_random(&layer.mlp_fc1, &rng);
        assign_random(&layer.mlp_fc2, &rng);
    }
    return host_model;
}

void load_model_from_f32(Model& host_model, const std::vector<float>& values) {
    std::size_t cursor = 0;
    auto load_into = [&](std::vector<double>* target) {
        if (cursor + target->size() > values.size()) {
            throw std::runtime_error("weights file is smaller than expected");
        }
        for (double& value : *target) {
            value = static_cast<double>(values[cursor++]);
        }
    };

    load_into(&host_model.wte);
    load_into(&host_model.wpe);
    load_into(&host_model.lm_head);
    for (LayerWeights& layer : host_model.layers) {
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

std::vector<double> flatten_model_values(const Model& host_model) {
    std::vector<double> values;
    std::size_t total_size = host_model.wte.size() + host_model.wpe.size() + host_model.lm_head.size();
    for (const LayerWeights& layer : host_model.layers) {
        total_size += layer.attn_wq.size() + layer.attn_wk.size() + layer.attn_wv.size() + layer.attn_wo.size() +
                      layer.mlp_fc1.size() + layer.mlp_fc2.size();
    }
    values.reserve(total_size);
    append_values(&values, host_model.wte);
    append_values(&values, host_model.wpe);
    append_values(&values, host_model.lm_head);
    for (const LayerWeights& layer : host_model.layers) {
        append_values(&values, layer.attn_wq);
        append_values(&values, layer.attn_wk);
        append_values(&values, layer.attn_wv);
        append_values(&values, layer.attn_wo);
        append_values(&values, layer.mlp_fc1);
        append_values(&values, layer.mlp_fc2);
    }
    return values;
}

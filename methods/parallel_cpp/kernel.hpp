#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

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

struct KernelResult {
  int seq_len = 0;
  std::vector<double> logits;
  double loss = 0.0;
};

struct BatchTokens {
  std::vector<int> tokens;
  // Flattened token IDs for the entire batch.
  // Layout: [batch_size, max_seq_len] stored row-major.
  // Access: tokens[b * max_seq_len + t]
  // Each value is a token index into the vocabulary.

  std::vector<int> seq_lens;
  // Length of each sequence INCLUDING the final target token.
  // Size = batch_size.
  // seq_lens[b] = number of valid tokens in sequence b.
  // Used to know which positions are real vs padding (future use).

  int batch_size = 0;
  // Number of sequences in the batch (B).

  int max_seq_len = 0;
  // Maximum sequence length across the batch (T_max).
  // Each sequence is padded to this length in `tokens`.
};

template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;
  std::size_t count = 0;
};

struct DeviceModel {
  ModelConfig config;
  DeviceBuffer<double> wte;
  DeviceBuffer<double> wpe;
  DeviceBuffer<double> lm_head;
  std::vector<DeviceBuffer<double>> attn_wq;
  std::vector<DeviceBuffer<double>> attn_wk;
  std::vector<DeviceBuffer<double>> attn_wv;
  std::vector<DeviceBuffer<double>> attn_wo;
  std::vector<DeviceBuffer<double>> mlp_fc1;
  std::vector<DeviceBuffer<double>> mlp_fc2;
};

struct DeviceWorkspace {
  DeviceBuffer<int> tokens;
  DeviceBuffer<int> seq_lens;
  DeviceBuffer<double> embeddings;
  DeviceBuffer<double> embeddings_output; 
  DeviceBuffer<double> hidden;
  // Forward-only transformer workspace and caches for future CUDA kernels.
  DeviceBuffer<double> x;
  DeviceBuffer<double> x_tmp;
  DeviceBuffer<double> norm;
  DeviceBuffer<double> q;
  DeviceBuffer<double> k_cache;
  DeviceBuffer<double> v_cache;
  DeviceBuffer<double> attn_out;
  DeviceBuffer<double> mlp_hidden;
  DeviceBuffer<double> logits;
  DeviceBuffer<double> loss;
};

struct KernelLaunch {
  int threads = 256;
  int blocks = 1;
};

inline int outline_usable_seq_len(const ModelConfig &config,
                                  const BatchTokens &batch) {
  return std::min(config.block_size, batch.max_seq_len - 1);
}

inline KernelLaunch make_1d_launch(std::size_t work_items, int threads = 256) {
  KernelLaunch shape;
  shape.threads = threads;
  shape.blocks =
      static_cast<int>((work_items + static_cast<std::size_t>(threads) - 1) /
                       static_cast<std::size_t>(threads));
  if (shape.blocks < 1) {
    shape.blocks = 1;
  }
  return shape;
}

Model make_empty_model(const ModelConfig &config);
Model initialize_model(const ModelConfig &config, std::uint32_t seed);
void load_model_from_f32(Model &host_model, const std::vector<float> &values);
std::vector<double> flatten_model_values(const Model &host_model);
DeviceModel upload_model_to_device(const Model &host_model);
void free_device_model(DeviceModel *device_model);
KernelResult run_forward_batched(const DeviceModel &device_model,
                                 const BatchTokens &batch);

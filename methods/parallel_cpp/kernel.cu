#include "kernel.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA failure while ") + action + ": " + cudaGetErrorString(status));
    }
}

void free_model(DeviceModel* device_model);
void free_workspace(DeviceWorkspace* workspace);

template <typename T>
void free_buffer(DeviceBuffer<T>* device_buffer);

std::size_t model_value_count(const ModelConfig& config) {
    std::size_t total_size = 0;
    total_size += static_cast<std::size_t>(config.vocab_size) * static_cast<std::size_t>(config.n_embd);   // wte
    total_size += static_cast<std::size_t>(config.block_size) * static_cast<std::size_t>(config.n_embd);   // wpe
    total_size += static_cast<std::size_t>(config.vocab_size) * static_cast<std::size_t>(config.n_embd);   // lm_head

    const std::size_t square = static_cast<std::size_t>(config.n_embd) * static_cast<std::size_t>(config.n_embd);
    const std::size_t mlp = static_cast<std::size_t>(config.mlp_dim()) * static_cast<std::size_t>(config.n_embd);
    total_size += static_cast<std::size_t>(config.n_layer) * (4 * square + 2 * mlp);
    return total_size;
}

void validate_batch(const BatchTokens& batch) {
    if (batch.batch_size < 1) {
        throw std::runtime_error("batch must contain at least one sequence");
    }
    if (static_cast<int>(batch.seq_lens.size()) != batch.batch_size) {
        throw std::runtime_error("batch seq_lens size must match batch_size");
    }
    if (batch.max_seq_len < 2) {
        throw std::runtime_error("token sequence must contain at least one input and one target token");
    }

    // TODO: add padded batching support. For now we assume all sequences share
    // the same max_seq_len and start with batch_size = 1.
    for (int seq_len_with_targets : batch.seq_lens) {
        if (seq_len_with_targets != batch.max_seq_len) {
            throw std::runtime_error("parallel_cpp batching outline does not support padding yet");
        }
    }
}

template <typename T>
void allocate_buffer(DeviceBuffer<T>* device_buffer, std::size_t count, bool zero_initialize = false) {
    device_buffer->ptr = nullptr;
    device_buffer->count = count;
    if (device_buffer->count == 0) {
        return;
    }
    cuda_check(
        cudaMalloc(reinterpret_cast<void**>(&device_buffer->ptr), device_buffer->count * sizeof(T)),
        "allocating device buffer"
    );
    if (zero_initialize) {
        cuda_check(
            cudaMemset(device_buffer->ptr, 0, device_buffer->count * sizeof(T)),
            "zeroing device buffer"
        );
    }
}

template <typename T>
void upload_buffer(DeviceBuffer<T>* device_buffer, const std::vector<T>& host_values) {
    allocate_buffer(device_buffer, host_values.size());
    if (device_buffer->count == 0) {
        return;
    }
    cuda_check(
        cudaMemcpy(device_buffer->ptr, host_values.data(), device_buffer->count * sizeof(T), cudaMemcpyHostToDevice),
        "copying host buffer to device"
    );
}

template <typename T>
void free_buffer(DeviceBuffer<T>* device_buffer) {
    if (device_buffer->ptr != nullptr) {
        cudaFree(device_buffer->ptr);
    }
    device_buffer->ptr = nullptr;
    device_buffer->count = 0;
}

DeviceModel upload_model(const Model& host_model) {
    DeviceModel device_model;
    device_model.config = host_model.config;
    try {
        upload_buffer(&device_model.wte, host_model.wte);
        upload_buffer(&device_model.wpe, host_model.wpe);
        upload_buffer(&device_model.lm_head, host_model.lm_head);

        device_model.attn_wq.reserve(host_model.layers.size());
        device_model.attn_wk.reserve(host_model.layers.size());
        device_model.attn_wv.reserve(host_model.layers.size());
        device_model.attn_wo.reserve(host_model.layers.size());
        device_model.mlp_fc1.reserve(host_model.layers.size());
        device_model.mlp_fc2.reserve(host_model.layers.size());

        for (const LayerWeights& layer : host_model.layers) {
            device_model.attn_wq.emplace_back();
            upload_buffer(&device_model.attn_wq.back(), layer.attn_wq);
            device_model.attn_wk.emplace_back();
            upload_buffer(&device_model.attn_wk.back(), layer.attn_wk);
            device_model.attn_wv.emplace_back();
            upload_buffer(&device_model.attn_wv.back(), layer.attn_wv);
            device_model.attn_wo.emplace_back();
            upload_buffer(&device_model.attn_wo.back(), layer.attn_wo);
            device_model.mlp_fc1.emplace_back();
            upload_buffer(&device_model.mlp_fc1.back(), layer.mlp_fc1);
            device_model.mlp_fc2.emplace_back();
            upload_buffer(&device_model.mlp_fc2.back(), layer.mlp_fc2);
        }
    } catch (...) {
        free_model(&device_model);
        throw;
    }
    return device_model;
}

void free_model(DeviceModel* device_model) {
    free_buffer(&device_model->wte);
    free_buffer(&device_model->wpe);
    free_buffer(&device_model->lm_head);

    for (DeviceBuffer<double>& buffer : device_model->attn_wq) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->attn_wk) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->attn_wv) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->attn_wo) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->mlp_fc1) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->mlp_fc2) {
        free_buffer(&buffer);
    }

    device_model->attn_wq.clear();
    device_model->attn_wk.clear();
    device_model->attn_wv.clear();
    device_model->attn_wo.clear();
    device_model->mlp_fc1.clear();
    device_model->mlp_fc2.clear();
}

DeviceWorkspace allocate_workspace(const ModelConfig& config, const BatchTokens& batch, int usable_seq_len) {
    DeviceWorkspace workspace;
    const std::size_t sequence_count = static_cast<std::size_t>(batch.batch_size);
    const std::size_t time_steps = static_cast<std::size_t>(usable_seq_len);
    const std::size_t hidden_count = sequence_count * time_steps * static_cast<std::size_t>(config.n_embd);
    const std::size_t logits_count = sequence_count * time_steps * static_cast<std::size_t>(config.vocab_size);
    const std::size_t grad_count = model_value_count(config);

    try {
        upload_buffer(&workspace.tokens, batch.tokens);
        upload_buffer(&workspace.seq_lens, batch.seq_lens);
        allocate_buffer(&workspace.embeddings, hidden_count, true);
        allocate_buffer(&workspace.hidden, hidden_count, true);
        allocate_buffer(&workspace.logits, logits_count, true);
        allocate_buffer(&workspace.loss, 1, true);
        allocate_buffer(&workspace.grads, grad_count, true);
    } catch (...) {
        free_workspace(&workspace);
        throw;
    }
    return workspace;
}

void free_workspace(DeviceWorkspace* workspace) {
    free_buffer(&workspace->tokens);
    free_buffer(&workspace->seq_lens);
    free_buffer(&workspace->embeddings);
    free_buffer(&workspace->hidden);
    free_buffer(&workspace->logits);
    free_buffer(&workspace->loss);
    free_buffer(&workspace->grads);
}

__global__ void embedding_lookup_kernel_outline(
    const int* tokens,
    const double* wte,
    const double* wpe,
    double* embeddings,
    int batch_size,
    int max_seq_len,
    int usable_seq_len,
    int n_embd
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * usable_seq_len * n_embd;
    if (idx >= total) {
        return;
    }

    const int col = idx % n_embd;
    const int token_slot = idx / n_embd;
    const int pos = token_slot % usable_seq_len;
    const int batch_idx = token_slot / usable_seq_len;
    const int token_idx = batch_idx * max_seq_len + pos;
    const int token_id = tokens[token_idx];

    const double token_val = wte[token_id * n_embd + col];
    const double pos_val = wpe[pos * n_embd + col];
    embeddings[idx] = token_val + pos_val;
}

__global__ void transformer_layer_kernel_outline(
    const double* layer_input,
    double* layer_output,
    const int* seq_lens,
    int batch_size,
    int max_seq_len,
    int n_embd,
    int head_dim,
    int n_head
) {
    // TODO:
    // - decide whether one kernel handles the whole layer or whether QKV, attention,
    //   output projection, and MLP should be separate launches
    // - read from layer_input[b, t, :]
    // - write the next hidden state into layer_output[b, t, :]
    // - keep attention confined to the same batch item b
    // - TODO: skip padded positions once padding is added
    (void)layer_input;
    (void)layer_output;
    (void)seq_lens;
    (void)batch_size;
    (void)max_seq_len;
    (void)n_embd;
    (void)head_dim;
    (void)n_head;
}

__global__ void logits_and_loss_kernel_outline(
    const double* hidden,
    const int* tokens,
    const double* lm_head,
    double* logits,
    double* loss,
    const int* seq_lens,
    int batch_size,
    int max_seq_len,
    int n_embd,
    int vocab_size
) {
    // TODO:
    // - project hidden[b, t, :] into logits[b, t, :]
    // - compare against tokens[b, t + 1] for valid positions only
    // - reduce the per-position losses into one scalar
    (void)hidden;
    (void)tokens;
    (void)lm_head;
    (void)logits;
    (void)loss;
    (void)seq_lens;
    (void)batch_size;
    (void)max_seq_len;
    (void)n_embd;
    (void)vocab_size;
}

__global__ void backward_kernel_outline(
    const int* tokens,
    const double* hidden,
    const double* logits,
    double* grads,
    const int* seq_lens,
    int batch_size,
    int max_seq_len,
    int n_embd,
    int vocab_size
) {
    // TODO:
    // - choose how you want to store intermediate activations for backward
    // - write gradients into the flattened grads buffer
    // - keep indexing batch-ready as (b, t, ...)
    (void)tokens;
    (void)hidden;
    (void)logits;
    (void)grads;
    (void)seq_lens;
    (void)batch_size;
    (void)max_seq_len;
    (void)n_embd;
    (void)vocab_size;
}

void launch_embedding_outline(const DeviceModel& device_model, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    // Pseudocode:
    // const int usable_seq_len = outline_usable_seq_len(config, batch);
    // const auto launch = make_1d_launch(batch.batch_size * usable_seq_len * config.n_embd);
    // embedding_lookup_kernel_outline<<<launch.blocks, launch.threads>>>(
    //     workspace->tokens.ptr,
    //     device_model.wte.ptr,
    //     device_model.wpe.ptr,
    //     workspace->embeddings.ptr,
    //     batch.batch_size,
    //     batch.max_seq_len,
    //     usable_seq_len,
    //     config.n_embd);
    // cuda_check(cudaGetLastError(), "launching embedding_lookup_kernel_outline");
    //
    // First implementation goal:
    // - map one thread to one (batch, position, embedding_dim) element
    const int usable_seq_len = outline_usable_seq_len(config, batch);
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch.batch_size) * static_cast<std::size_t>(usable_seq_len) *
        static_cast<std::size_t>(config.n_embd)
    );
    embedding_lookup_kernel_outline<<<launch.blocks, launch.threads>>>(
        workspace->tokens.ptr,
        device_model.wte.ptr,
        device_model.wpe.ptr,
        workspace->embeddings.ptr,
        batch.batch_size,
        batch.max_seq_len,
        usable_seq_len,
        config.n_embd
    );
    cuda_check(cudaGetLastError(), "launching embedding_lookup_kernel_outline");
}

void launch_transformer_outline(const DeviceModel&, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    // Pseudocode:
    // for each transformer layer:
    //   choose either:
    //   A) one coarse kernel for the whole layer, or
    //   B) separate kernels for
    //      - norm
    //      - qkv projection
    //      - attention scores / softmax
    //      - attention output projection
    //      - mlp
    //
    // Example starting point for a coarse launch:
    // const auto launch = make_1d_launch(batch.batch_size * batch.max_seq_len * config.n_embd);
    // transformer_layer_kernel_outline<<<launch.blocks, launch.threads>>>(...);
    // cuda_check(cudaGetLastError(), "launching transformer_layer_kernel_outline");
    (void)workspace;
    (void)config;
    (void)batch;
}

void launch_logits_and_loss_outline(const DeviceModel& device_model, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    // Pseudocode:
    // 1. Project hidden[b, t, :] -> logits[b, t, :]
    // 2. Compute per-position cross-entropy against tokens[b, t + 1]
    // 3. Reduce into one scalar loss
    //
    // You can start with one kernel for logits and one kernel for the loss reduction.
    (void)device_model;
    (void)workspace;
    (void)config;
    (void)batch;
}

void launch_backward_outline(DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    // Pseudocode:
    // - decide what forward intermediates backward needs
    // - either:
    //   A) one coarse backward kernel, or
    //   B) a sequence of backward kernels mirroring the forward stages
    // - write gradients into the flattened workspace->grads buffer
    (void)workspace;
    (void)config;
    (void)batch;
}

}  // namespace

DeviceModel upload_model_to_device(const Model& host_model) {
    return upload_model(host_model);
}

void free_device_model(DeviceModel* device_model) {
    if (device_model == nullptr) {
        return;
    }
    free_model(device_model);
    device_model->config = ModelConfig{};
}

KernelResult run_forward_backward_batched(const DeviceModel& device_model, const BatchTokens& batch) {
    validate_batch(batch);
    const int usable_seq_len = outline_usable_seq_len(device_model.config, batch);

    // Host-side pseudocode for the full CUDA path:
    //
    DeviceWorkspace workspace = allocate_workspace(device_model.config, batch, usable_seq_len);
    try {
           launch_embedding_outline(device_model, &workspace, device_model.config, batch);
    //     launch_transformer_outline(device_model, &workspace, device_model.config, batch);
    //     launch_logits_and_loss_outline(device_model, &workspace, device_model.config, batch);
    //     launch_backward_outline(&workspace, device_model.config, batch);
    //     cuda_check(cudaDeviceSynchronize(), "synchronizing CUDA kernels");
    //     // copy logits / loss / grads back to host
    //     // pack a KernelResult and return it
        throw std::runtime_error(
            "parallel_cpp CUDA outline launched placeholder kernels only. Fill in methods/parallel_cpp/kernel.cu to produce logits, loss, and gradients."
        );
    } catch (...) {
        free_workspace(&workspace);
        throw;
    }
}

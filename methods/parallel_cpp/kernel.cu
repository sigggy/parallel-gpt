#include "kernel.hpp"

#include <cuda_runtime.h>

#include <driver_types.h>
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

void validate_batch(const BatchTokens& batch) {
    if (batch.batch_size < 1) {
        throw std::runtime_error("batch must contain at least one sequence");
    }
    if (batch.batch_seq_length < 2) {
        throw std::runtime_error("token sequence must contain at least one input and one target token");
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
    const std::size_t kv_cache_count = static_cast<std::size_t>(config.n_layer) * hidden_count;
    const std::size_t mlp_hidden_count = sequence_count * time_steps * static_cast<std::size_t>(config.mlp_dim());
    const std::size_t logits_count = sequence_count * time_steps * static_cast<std::size_t>(config.vocab_size);

    try {
        upload_buffer(&workspace.tokens, batch.tokens);
        allocate_buffer(&workspace.embeddings, hidden_count, true);
        allocate_buffer(&workspace.hidden, hidden_count, true);
        allocate_buffer(&workspace.x, hidden_count, true);
        allocate_buffer(&workspace.x_tmp, hidden_count, true);
        allocate_buffer(&workspace.norm, hidden_count, true);
        allocate_buffer(&workspace.q, hidden_count, true);
        allocate_buffer(&workspace.k_cache, kv_cache_count, true);
        allocate_buffer(&workspace.v_cache, kv_cache_count, true);
        allocate_buffer(&workspace.attn_out, hidden_count, true);
        allocate_buffer(&workspace.mlp_hidden, mlp_hidden_count, true);
        allocate_buffer(&workspace.logits, logits_count, true);
        allocate_buffer(&workspace.loss, 1, true);
    } catch (...) {
        free_workspace(&workspace);
        throw;
    }
    return workspace;
}

void free_workspace(DeviceWorkspace* workspace) {
    free_buffer(&workspace->tokens);
    free_buffer(&workspace->embeddings);
    free_buffer(&workspace->hidden);
    free_buffer(&workspace->x);
    free_buffer(&workspace->x_tmp);
    free_buffer(&workspace->norm);
    free_buffer(&workspace->q);
    free_buffer(&workspace->k_cache);
    free_buffer(&workspace->v_cache);
    free_buffer(&workspace->attn_out);
    free_buffer(&workspace->mlp_hidden);
    free_buffer(&workspace->logits);
    free_buffer(&workspace->loss);
}

__global__ void embedding_lookup_kernel(
    const int* tokens,
    const double* wte,
    const double* wpe,
    double* embeddings,
    int batch_size,
    int batch_seq_length,
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
    const int token_idx = batch_idx * batch_seq_length + pos;
    const int token_id = tokens[token_idx];

    const double token_val = wte[token_id * n_embd + col];
    const double pos_val = wpe[pos * n_embd + col];
    embeddings[idx] = token_val + pos_val;
}

__global__ void rmsnorm_kernel(
    const double* input,
    double* output,
    int n_embd,
    int useable_seq_len, 
    int num_batches
) {
    int total_tokens = num_batches * useable_seq_len;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= total_tokens) return;

    int start = idx * n_embd;

    double mean_square = 0.0;

    for (int i = 0; i < n_embd; i++) {
        double value = input[start + i];
        mean_square += value * value;
    }

    mean_square /= (double)n_embd;

    double scale = 1.0 / sqrt(mean_square + 1e-5);

    for (int i = 0; i < n_embd; i++) {
        output[start + i] = input[start + i] * scale;
    }
}


__global__ void linear(
    const double* input,
    double* output,
    const double* weights,
    int in_dim,
    int out_dim,
    int num_batches,
    int usable_seq_len
) {
    int total_tokens = num_batches * usable_seq_len;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= total_tokens) return;

    int input_start = idx * in_dim;
    int output_start = idx * out_dim;

    for (int out = 0; out < out_dim; ++out) {
        double sum = 0.0;

        for (int in = 0; in < in_dim; ++in) {
            sum += weights[out * in_dim + in] * input[input_start + in];
        }

        output[output_start + out] = sum;
    }
}


__global__ void logits_and_loss_kernel_outline(
    const double* hidden,
    const int* tokens,
    const double* lm_head,
    double* logits,
    double* loss,
    int batch_size,
    int batch_seq_length,
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
    (void)batch_size;
    (void)batch_seq_length;
    (void)n_embd;
    (void)vocab_size;
}

void launch_embedding(const DeviceModel& device_model, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    const int usable_seq_len = outline_usable_seq_len(config, batch);
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch.batch_size) * static_cast<std::size_t>(usable_seq_len) *
        static_cast<std::size_t>(config.n_embd)
    );
    embedding_lookup_kernel<<<launch.blocks, launch.threads>>>(
        workspace->tokens.ptr,
        device_model.wte.ptr,
        device_model.wpe.ptr,
        workspace->embeddings.ptr,
        batch.batch_size,
        batch.batch_seq_length,
        usable_seq_len,
        config.n_embd
    );
    cuda_check(cudaGetLastError(), "launching embedding_lookup_kernel_outline");
}


void launch_rmsnorm(
    const double* input,
    double* output,
    int n_embd,
    int batch_size,
    int usable_seq_len
) {
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(usable_seq_len)
    );

    rmsnorm_kernel<<<launch.blocks, launch.threads>>>(
        input,
        output,
        n_embd,
        usable_seq_len,
        batch_size
    );

    cuda_check(cudaGetLastError(), "launching rmsnorm_kernel");
}



void launch_transformer(const DeviceModel& device_model, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    /*
    embedding_lookup -> workspace.x

    for each layer:
        rmsnorm(x -> norm)
        linear(norm -> q)
        linear(norm -> k_cache[layer])
        linear(norm -> v_cache[layer])
        attention(q, k_cache[layer], v_cache[layer] -> attn_out)
        linear(attn_out -> x_tmp)
        residual_add(x, x_tmp -> x)

        rmsnorm(x -> norm)
        linear(norm -> mlp_hidden)
        relu(mlp_hidden)
        linear(mlp_hidden -> x_tmp)
        residual_add(x, x_tmp -> x)

    logits_and_loss(x -> logits/loss)
    */

    const int usable_seq_len = outline_usable_seq_len(config, batch);
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch.batch_size) * static_cast<std::size_t>(usable_seq_len) *
        static_cast<std::size_t>(config.n_embd)
    );

    launch_embedding(device_model, workspace, config, batch);
    launch_rmsnorm(workspace->embeddings.ptr, workspace->x.ptr, config.n_embd, batch.batch_size, usable_seq_len);

    for (int layer_idx = 0; layer_idx < config.n_layer; ++layer_idx) {
        launch_rmsnorm(workspace->x.ptr, workspace->norm.ptr, config.n_embd, batch.batch_size, usable_seq_len);



    }
    
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

KernelResult run_forward_batched(const DeviceModel& device_model, const BatchTokens& batch) {
    validate_batch(batch);
    const int usable_seq_len = outline_usable_seq_len(device_model.config, batch);

    // Host-side pseudocode for the full CUDA path:
    //
    DeviceWorkspace workspace = allocate_workspace(device_model.config, batch, usable_seq_len);
    try {
           launch_transformer(device_model, &workspace, device_model.config, batch);
    //     launch_logits_and_loss_outline(device_model, &workspace, device_model.config, batch);
    //     cuda_check(cudaDeviceSynchronize(), "synchronizing CUDA kernels");
    //     // copy logits / loss back to host
    //     // pack a KernelResult and return it
        throw std::runtime_error(
            "parallel_cpp CUDA outline launched placeholder kernels only. Fill in methods/parallel_cpp/kernel.cu to produce logits and loss."
        );
    } catch (...) {
        free_workspace(&workspace);
        throw;
    }
}

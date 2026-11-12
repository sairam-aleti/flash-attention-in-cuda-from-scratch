"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // 1. Calculate the global index for this specific thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Ensure threads outside the array bounds do nothing
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        a[idx] *= scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] = expf(a[i]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float max_val = matrix[r * cols];

        for (int c = 1; c < cols; ++c) {
            float val = matrix[r * cols + c];
            if (val > max_val) {
                max_val = val;
            }
        }

        out[r] = max_val;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float sum = 0.0f;

        for (int c = 0; c < cols; ++c) {
            sum += matrix[r * cols + c];
        }

        out[r] = sum;
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c,
                       int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;

        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }

        c[row * n + col] = sum;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k,
                          float* scores, int seq_len, int head_dim) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {

        const float* q_row = q + i * head_dim;
        const float* k_row = k + j * head_dim;

        scores[i * seq_len + j] =
            dot_product(q_row, k_row, head_dim) / sqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    int offset = row * cols;

    // Step 1: Find row maximum
    float local_max = -INFINITY;

    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, matrix[offset + i]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = sdata[0];
    __syncthreads();

    // Step 2: Compute exponentials and local sum
    float local_sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(matrix[offset + i] - row_max);
        matrix[offset + i] = val;
        local_sum += val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Step 3: Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = sdata[0];
    __syncthreads();

    // Step 4: Normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        matrix[offset + i] /= row_sum;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;

        for (int j = 0; j < seq_len; j++) {
            sum += p[i * seq_len + j] * v[j * head_dim + d];
        }

        out[i * head_dim + d] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k,
                     const float* d_v, float* d_out,
                     int seq_len, int head_dim) {
    float* d_scores;

    // Allocate intermediate attention score matrix
    cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));

    // Stage 1: Compute Q * K^T
    dim3 block1(16, 16);
    dim3 grid1((seq_len + 15) / 16, (seq_len + 15) / 16);

    qk_scores<<<grid1, block1>>>(
        d_q, d_k, d_scores, seq_len, head_dim
    );

    // Stage 2: Apply row-wise softmax
    int threads = 128;
    int shared_mem = threads * sizeof(float);

    softmax_rows<<<seq_len, threads, shared_mem>>>(
        d_scores, seq_len, seq_len
    );

    // Stage 3: Compute P * V
    dim3 block3(16, 16);
    dim3 grid3((head_dim + 15) / 16, (seq_len + 15) / 16);

    pv_matmul<<<grid3, block3>>>(
        d_scores, d_v, d_out, seq_len, head_dim
    );

    // Free intermediate buffer
    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return old_sum * correction + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int i = 0; i < head_dim; i++) {
        out_row[i] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    int total_elements = tile_rows * tile_cols;

    // Threads cooperatively load elements in a strided manner.
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int tile_row = idx / tile_cols;
        int tile_col = idx % tile_cols;

        int src_row = src_row_start + tile_row;
        int src_col = src_col_start + tile_col;

        // Check source matrix bounds.
        if (src_row >= 0 && src_row < src_rows &&
            src_col >= 0 && src_col < src_cols) {
            shared_dst[idx] = src[src_row * src_cols + src_col];
        } else {
            // Zero-fill out-of-bounds tile positions.
            shared_dst[idx] = 0.0f;
        }
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    int total_elements = tile_q * tile_k;

    // Threads cooperatively compute output scores.
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;

        float dot = 0.0f;

        // Compute dot product of query row i and key row j.
        for (int d = 0; d < head_dim; ++d) {
            dot += q_tile[i * head_dim + d] *
                   k_tile[j * head_dim + d];
        }

        // Apply scaling and store the score.
        s_tile[idx] = dot * scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile,
                            float* row_max_out,
                            int tile_q,
                            int tile_k,
                            int thread_id,
                            int num_threads) {
    // Threads cooperatively process rows.
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float max_val = s_tile[r * tile_k];

        // Find maximum value across the row.
        for (int c = 1; c < tile_k; ++c) {
            float val = s_tile[r * tile_k + c];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Store the row maximum.
        row_max_out[r] = max_val;
    }
}

# Step 20 - tile_exp
#include <cmath>

__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int total = tile_q * tile_k;

    for (int idx = thread_id; idx < total; idx += num_threads) {
        int r = idx / tile_k;

        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float sum = 0.0f;

        for (int k = 0; k < tile_k; ++k) {
            sum += p_tile[r * tile_k + k];
        }

        row_sum_out[r] = sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile,
                              const float* v_tile,
                              float* out_acc,
                              int tile_q,
                              int tile_k,
                              int head_dim,
                              int thread_id,
                              int num_threads) {
    int total = tile_q * head_dim;

    for (int idx = thread_id; idx < total; idx += num_threads) {
        int q = idx / head_dim;
        int d = idx % head_dim;

        float sum = 0.0f;

        for (int k = 0; k < tile_k; ++k) {
            sum += p_tile[q * tile_k + k] *
                   v_tile[k * head_dim + d];
        }

        out_acc[idx] += sum;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k,
                                       const float* v, float* out,
                                       int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int nt = blockDim.x;

    int q_start = blockIdx.x * tile_q;
    int q_rows = min(tile_q, seq_len - q_start);

    // Shared memory layout.
    float* qt = smem;
    float* kt = qt + tile_q * head_dim;
    float* vt = kt + tile_k * head_dim;
    float* scores = vt + tile_k * head_dim;
    float* acc = scores + tile_q * tile_k;

    float* m = acc + tile_q * head_dim;
    float* l = m + tile_q;
    float* mb = l + tile_q;
    float* lb = mb + tile_q;

    // Initialize accumulator and online softmax statistics.
    for (int i = tid; i < tile_q * head_dim; i += nt)
        acc[i] = 0.0f;

    for (int i = tid; i < tile_q; i += nt) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        mb[i] = -INFINITY;
        lb[i] = 0.0f;
    }

    __syncthreads();

    // Load Q tile ONCE.
    load_tile(q, qt,
              q_start, 0,
              seq_len, head_dim,
              tile_q, head_dim,
              tid, nt);

    __syncthreads();

    // Stream through K/V tiles.
    for (int ks = 0; ks < seq_len; ks += tile_k) {

        int kr = min(tile_k, seq_len - ks);

        load_tile(k, kt,
                  ks, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  tid, nt);

        load_tile(v, vt,
                  ks, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  tid, nt);

        __syncthreads();

        // Compute QK^T.
        tile_scores(qt, kt, scores,
                    tile_q, tile_k, head_dim,
                    scale, tid, nt);

        __syncthreads();

        // Mask invalid query and key positions.
        for (int i = tid; i < tile_q * tile_k; i += nt) {
            int r = i / tile_k;
            int c = i % tile_k;

            if (r >= q_rows || c >= kr)
                scores[i] = -INFINITY;
        }

        __syncthreads();

        // Find maximum score in current tile.
        tile_rowmax(scores, mb,
                    tile_q, tile_k, tid, nt);

        __syncthreads();

        // Update running maximum and rescale previous results.
        for (int r = tid; r < tile_q; r += nt) {

            float old_max = m[r];
            float new_max = online_max(old_max, mb[r]);

            float correction =
                correction_factor(old_max, new_max);

            lb[r] = correction;

            rescale_output(acc + r * head_dim,
                           head_dim, correction);

            l[r] *= correction;

            m[r] = new_max;
        }

        __syncthreads();

        // Exponentiate using updated running maximum.
        tile_exp(scores, m,
                 tile_q, tile_k, tid, nt);

        __syncthreads();

        // Sum exponentials for current tile.
        tile_rowsum(scores, mb,
                    tile_q, tile_k, tid, nt);

        __syncthreads();

        // Update running denominator.
        for (int r = tid; r < tile_q; r += nt) {
            l[r] = update_running_sum(
                l[r], 1.0f, mb[r]
            );
        }

        __syncthreads();

        // Accumulate weighted values.
        accumulate_pv(scores, vt, acc,
                      tile_q, tile_k, head_dim,
                      tid, nt);

        __syncthreads();
    }

    // Normalize and write output.
    for (int i = tid; i < q_rows * head_dim; i += nt) {

        int r = i / head_dim;
        int d = i % head_dim;

        out[(q_start + r) * head_dim + d] =
            l[r] > 0.0f
            ? acc[r * head_dim + d] / l[r]
            : 0.0f;
    }
}

# Step 24 - flash_attention_launcher
void flash_attention_launcher(const float* d_q, const float* d_k,
                              const float* d_v, float* d_out,
                              int seq_len, int head_dim,
                              int tile_q, int tile_k) {

    // One block per query tile.
    int blocks = (seq_len + tile_q - 1) / tile_q;

    // 128 threads per block.
    int threads = 128;

    // Calculate scaling factor.
    float scale = 1.0f / sqrtf((float)head_dim);

    // Calculate dynamic shared memory.
    size_t smem_elements =
        tile_q * head_dim +       // Q tile
        tile_k * head_dim +       // K tile
        tile_k * head_dim +       // V tile
        tile_q * tile_k +         // Score tile
        tile_q * head_dim +       // Output accumulator
        4 * tile_q;               // Running statistics

    size_t smem_bytes = smem_elements * sizeof(float);

    // Launch Flash Attention kernel.
    flash_attention_kernel<<<blocks, threads, smem_bytes>>>(
        d_q, d_k, d_v, d_out,
        seq_len, head_dim,
        tile_q, tile_k, scale
    );
}

# Step 25 - causal_mask
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {

    int total = tile_q * tile_k;

    for (int idx = thread_id; idx < total; idx += num_threads) {
        int q = idx / tile_k;
        int k = idx % tile_k;

        int global_q = q_row_start + q;
        int global_k = k_col_start + k;

        // Mask future key positions.
        if (global_k > global_q) {
            s_tile[idx] = -INFINITY;
        }
    }
}

# Step 26 - flash_attention_causal_kernel
__global__ void flash_attention_causal_kernel(
    const float* q, const float* k, const float* v,
    float* out, int seq_len, int head_dim,
    int tile_q, int tile_k, float scale) {

    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int nt = blockDim.x;

    int q_start = blockIdx.x * tile_q;
    int q_rows = min(tile_q, seq_len - q_start);

    // Shared memory layout.
    float* qt = smem;
    float* kt = qt + tile_q * head_dim;
    float* vt = kt + tile_k * head_dim;
    float* scores = vt + tile_k * head_dim;
    float* acc = scores + tile_q * tile_k;

    float* m = acc + tile_q * head_dim;
    float* l = m + tile_q;
    float* mb = l + tile_q;
    float* lb = mb + tile_q;

    // Initialize output accumulator and online softmax state.
    for (int i = tid; i < tile_q * head_dim; i += nt)
        acc[i] = 0.0f;

    for (int i = tid; i < tile_q; i += nt) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        mb[i] = -INFINITY;
        lb[i] = 0.0f;
    }

    __syncthreads();

    // Load query tile once.
    load_tile(q, qt,
              q_start, 0,
              seq_len, head_dim,
              tile_q, head_dim,
              tid, nt);

    __syncthreads();

    // Stream over key/value tiles.
    for (int ks = 0; ks < seq_len; ks += tile_k) {

        // Skip tiles entirely in the future.
        if (ks >= q_start + q_rows)
            break;

        int kr = min(tile_k, seq_len - ks);

        load_tile(k, kt,
                  ks, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  tid, nt);

        load_tile(v, vt,
                  ks, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  tid, nt);

        __syncthreads();

        // Compute Q * K^T.
        tile_scores(qt, kt, scores,
                    tile_q, tile_k, head_dim,
                    scale, tid, nt);

        __syncthreads();

        // Apply causal mask.
        causal_mask(scores,
                    q_start, ks,
                    tile_q, tile_k,
                    tid, nt);

        // Mask ragged query and key positions.
        for (int i = tid; i < tile_q * tile_k; i += nt) {
            int r = i / tile_k;
            int c = i % tile_k;

            if (r >= q_rows || c >= kr)
                scores[i] = -INFINITY;
        }

        __syncthreads();

        // Find maximum score for this tile.
        tile_rowmax(scores, mb,
                    tile_q, tile_k,
                    tid, nt);

        __syncthreads();

        // Update running maximum and rescale old output.
        for (int r = tid; r < tile_q; r += nt) {

            float old_max = m[r];
            float new_max = online_max(old_max, mb[r]);

            float correction =
                correction_factor(old_max, new_max);

            lb[r] = correction;

            rescale_output(acc + r * head_dim,
                           head_dim, correction);

            l[r] *= correction;

            m[r] = new_max;
        }

        __syncthreads();

        // Compute exponentials relative to updated maximum.
        tile_exp(scores, m,
                 tile_q, tile_k,
                 tid, nt);

        __syncthreads();

        // Compute current tile's row sums.
        tile_rowsum(scores, mb,
                    tile_q, tile_k,
                    tid, nt);

        __syncthreads();

        // Update running denominator.
        for (int r = tid; r < tile_q; r += nt) {
            l[r] = update_running_sum(
                l[r], 1.0f, mb[r]
            );
        }

        __syncthreads();

        // Accumulate weighted values.
        accumulate_pv(scores, vt, acc,
                      tile_q, tile_k, head_dim,
                      tid, nt);

        __syncthreads();
    }

    // Normalize and write final output.
    for (int i = tid; i < q_rows * head_dim; i += nt) {

        int r = i / head_dim;
        int d = i % head_dim;

        out[(q_start + r) * head_dim + d] =
            l[r] > 0.0f
            ? acc[r * head_dim + d] / l[r]
            : 0.0f;
    }
}


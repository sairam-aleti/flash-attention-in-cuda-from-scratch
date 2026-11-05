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

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement


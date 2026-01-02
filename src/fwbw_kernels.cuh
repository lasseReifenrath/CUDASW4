#ifndef FWBW_KERNELS_CUH
#define FWBW_KERNELS_CUH

#include "config.hpp"
#include "blosum.hpp"
#include "types.hpp"
#include "util.cuh"
#include <cuda_runtime.h>

namespace cudasw4{
namespace fwbw{

// Forward pass kernel: Compute Z^M, Z^E, Z^F matrices with normalization
// Uses wavefront scheduling: each warp processes one alignment,
// each thread processes K consecutive columns
template<int WARP_SIZE, int K, int BLOSUM_DIM>
__global__ void forward_pass_kernel(
    const char* queries,
    const char* targets,
    float* zm_matrix,              // Output: log(ZM) values [num_alignments × query_len × target_len]
    float* ze_row_global,          // Output: ZE row vector [num_alignments × target_len]
    float* zf_row_global,          // Output: ZF row vector [num_alignments × target_len]
    float* log_scales_fwd,         // Output: normalization factors [num_alignments × query_len]
    const size_t* query_offsets,
    const size_t* target_offsets,
    const SequenceLengthT* query_lengths,
    const SequenceLengthT* target_lengths,
    const float beta,
    const float gap_open,
    const float gap_extend,
    const int num_alignments
)
{
    // Warp and lane identification
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_alignments) return;

    // Load alignment parameters
    const int query_len = query_lengths[warp_id];
    const int target_len = target_lengths[warp_id];
    const size_t query_off = query_offsets[warp_id];
    const size_t target_off = target_offsets[warp_id];

    // Precompute constants
    const float exp_go = expf(beta * gap_open);
    const float exp_ge = expf(beta * gap_extend);

    // Thread's column range
    const int col_start = lane_id * K;
    const int col_end = min(col_start + K, target_len);
    const int num_cols = col_end - col_start;

    // Load target sequence into registers (reused for all rows)
    char target_chars[K];
    #pragma unroll
    for (int c = 0; c < K; c++) {
        int col = col_start + c;
        target_chars[c] = (col < target_len) ? targets[target_off + col] : 0;
    }

    // Initialize buffers for "previous row" (row -1)
    float ZM_prev[K];
    float ZE_prev[K];
    float ZF_prev[K];

    #pragma unroll
    for (int c = 0; c < K; c++) {
        ZM_prev[c] = 0.0f;
        ZE_prev[c] = 0.0f;
        ZF_prev[c] = 0.0f;
    }

    // Wavefront iterations
    const int num_iterations = query_len + WARP_SIZE - 1;

    for (int iter = 0; iter < num_iterations; iter++) {

        int row = iter - lane_id;

        // Check if this thread has valid work this iteration
        if (row >= 0 && row < query_len) {

            // Load query character for this row
            char query_char = queries[query_off + row];

            // Current row buffers
            float ZM_curr[K];
            float ZE_curr[K];
            float ZF_curr[K];

            // Precompute scores for this row
            float scores[K];
            float exp_scores[K];

            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                // Get score from BLOSUM matrix (stored in constant memory)
                scores[c] = deviceBlosum[int(query_char) * BLOSUM_DIM + int(target_chars[c])];
                exp_scores[c] = expf(beta * scores[c]);
            }

            // ===== COMPUTE M AND F MATRICES =====
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {

                // Get diagonal values [i-1, j-1] for ZM
                float ZM_diag, ZE_diag, ZF_diag;

                if (c == 0) {
                    // First column in thread - need from previous thread
                    if (lane_id == 0) {
                        // Boundary condition
                        ZM_diag = (row == 0 && col_start == 0) ? 1.0f : 0.0f;
                        ZE_diag = 0.0f;
                        ZF_diag = 0.0f;
                    } else {
                        // Get last cell from previous thread's previous row
                        ZM_diag = __shfl_up_sync(0xFFFFFFFF, ZM_prev[K-1], 1);
                        ZE_diag = __shfl_up_sync(0xFFFFFFFF, ZE_prev[K-1], 1);
                        ZF_diag = __shfl_up_sync(0xFFFFFFFF, ZF_prev[K-1], 1);
                    }
                } else {
                    // Diagonal from same thread, previous row
                    ZM_diag = ZM_prev[c-1];
                    ZE_diag = ZE_prev[c-1];
                    ZF_diag = ZF_prev[c-1];
                }

                // Compute Z^M[i,j] = (Z^M[i-1,j-1] + Z^E[i-1,j-1] + Z^F[i-1,j-1]) * exp(beta * S[i,j])
                ZM_curr[c] = (ZM_diag + ZE_diag + ZF_diag) * exp_scores[c];

                // Get top values [i-1, j] for ZF
                float ZM_top = ZM_prev[c];
                float ZF_top = ZF_prev[c];

                // Compute Z^F[i,j] = Z^M[i-1,j] * exp(beta * gap_open) + Z^F[i-1,j] * exp(beta * gap_extend)
                ZF_curr[c] = ZM_top * exp_go + ZF_top * exp_ge;
            }

            // ===== COMPUTE E MATRIX (SEQUENTIAL - NO PREFIX SUM NEEDED!) =====
            // This is the key insight: in CUDA, each thread processes sequentially,
            // so E-matrix dependencies are naturally resolved. No prefix sum needed!

            #pragma unroll
            for (int c = 0; c < num_cols; c++) {

                // Get left values [i, j-1]
                float ZM_left, ZE_left;

                if (c == 0) {
                    // First column - need from previous thread (same row)
                    if (lane_id == 0) {
                        ZM_left = 0.0f;
                        ZE_left = 0.0f;
                    } else {
                        // Get from previous thread's last computed cell
                        ZM_left = __shfl_up_sync(0xFFFFFFFF, ZM_curr[K-1], 1);
                        ZE_left = __shfl_up_sync(0xFFFFFFFF, ZE_curr[K-1], 1);
                    }
                } else {
                    // Already computed in this thread
                    ZM_left = ZM_curr[c-1];
                    ZE_left = ZE_curr[c-1];
                }

                // Compute Z^E[i,j] = Z^M[i,j-1] * exp(beta * gap_open) + Z^E[i,j-1] * exp(beta * gap_extend)
                ZE_curr[c] = ZM_left * exp_go + ZE_left * exp_ge;
            }

            // ===== NORMALIZATION =====
            // Find max value in this row
            float max_val = 0.0f;
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                max_val = fmaxf(max_val, ZM_curr[c]);
                max_val = fmaxf(max_val, ZE_curr[c]);
                max_val = fmaxf(max_val, ZF_curr[c]);
            }

            // Warp reduction to find global max for this row
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
            }
            max_val = __shfl_sync(0xFFFFFFFF, max_val, 0); // Broadcast to all lanes

            // Normalize all values
            if (max_val > 1e-30f) {
                const float inv_max = 1.0f / max_val;
                #pragma unroll
                for (int c = 0; c < num_cols; c++) {
                    ZM_curr[c] *= inv_max;
                    ZE_curr[c] *= inv_max;
                    ZF_curr[c] *= inv_max;
                }

                // Store log scale factor (only one thread per row needs to write)
                if (lane_id == 0) {
                    log_scales_fwd[warp_id * query_len + row] = logf(max_val);
                }
            } else {
                if (lane_id == 0) {
                    log_scales_fwd[warp_id * query_len + row] = -1e30f; // log(0) = -inf
                }
            }

            // ===== WRITE TO GLOBAL MEMORY =====

            // Write ZM to main matrix (in log space)
            const size_t matrix_offset = size_t(warp_id) * query_len * target_len + size_t(row) * target_len;
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                int col = col_start + c;
                zm_matrix[matrix_offset + col] = logf(ZM_curr[c] + 1e-30f);
            }

            // Write ZE and ZF to row vectors (for next row computation)
            // These will be read by this thread in next iteration as ZE_prev, ZF_prev
            const size_t vector_offset = size_t(warp_id) * target_len;
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                int col = col_start + c;
                ze_row_global[vector_offset + col] = ZE_curr[c];
                zf_row_global[vector_offset + col] = ZF_curr[c];
            }

            // ===== UPDATE BUFFERS FOR NEXT ITERATION =====
            #pragma unroll
            for (int c = 0; c < K; c++) {
                ZM_prev[c] = (c < num_cols) ? ZM_curr[c] : 0.0f;
                ZE_prev[c] = (c < num_cols) ? ZE_curr[c] : 0.0f;
                ZF_prev[c] = (c < num_cols) ? ZF_curr[c] : 0.0f;
            }

        } // if valid row

    } // for each iteration
}


// Backward pass kernel: Similar to forward, but iterate in reverse
// Computes Z^M_bwd, Z^E_bwd, Z^F_bwd from bottom-right to top-left
template<int WARP_SIZE, int K, int BLOSUM_DIM>
__global__ void backward_pass_kernel(
    const char* queries,
    const char* targets,
    const float* zm_matrix_fwd,    // Input: from forward pass (unused here, for reference)
    float* zm_matrix_bwd,          // Output: log(ZM_backward)
    float* ze_row_global,          // Workspace: ZE row vector
    float* zf_row_global,          // Workspace: ZF row vector
    float* log_scales_bwd,         // Output: normalization factors
    const size_t* query_offsets,
    const size_t* target_offsets,
    const SequenceLengthT* query_lengths,
    const SequenceLengthT* target_lengths,
    const float beta,
    const float gap_open,
    const float gap_extend,
    const int num_alignments
)
{
    // Warp and lane identification
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_alignments) return;

    // Load alignment parameters
    const int query_len = query_lengths[warp_id];
    const int target_len = target_lengths[warp_id];
    const size_t query_off = query_offsets[warp_id];
    const size_t target_off = target_offsets[warp_id];

    // Precompute constants
    const float exp_go = expf(beta * gap_open);
    const float exp_ge = expf(beta * gap_extend);

    // Thread's column range (processing right-to-left, so reverse the mapping)
    // Lane 0 handles rightmost columns, lane 31 handles leftmost
    const int col_end_rev = target_len - 1 - lane_id * K;  // Rightmost column for this thread
    const int col_start_rev = max(col_end_rev - K + 1, 0); // Leftmost column for this thread
    const int num_cols = max(0, col_end_rev - col_start_rev + 1);

    // Load target sequence into registers (reversed order within thread)
    char target_chars[K];
    #pragma unroll
    for (int c = 0; c < K; c++) {
        int col = col_end_rev - c;  // Process right to left
        target_chars[c] = (col >= 0 && col < target_len) ? targets[target_off + col] : 0;
    }

    // Initialize buffers for "next row" (row query_len, which is boundary)
    float ZM_next[K];
    float ZE_next[K];
    float ZF_next[K];

    #pragma unroll
    for (int c = 0; c < K; c++) {
        ZM_next[c] = 0.0f;
        ZE_next[c] = 0.0f;
        ZF_next[c] = 0.0f;
    }

    // Wavefront iterations (reverse order)
    const int num_iterations = query_len + WARP_SIZE - 1;

    for (int iter = 0; iter < num_iterations; iter++) {

        // Compute row index (reversed: start from query_len-1, go down)
        int row = (query_len - 1) - (iter - (WARP_SIZE - 1 - lane_id));

        // Check if this thread has valid work this iteration
        if (row >= 0 && row < query_len) {

            // Load query character for this row
            char query_char = queries[query_off + row];

            // Current row buffers
            float ZM_curr[K];
            float ZE_curr[K];
            float ZF_curr[K];

            // Precompute scores for this row
            float scores[K];
            float exp_scores[K];

            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                int col = col_end_rev - c;
                char tc = target_chars[c];
                // Get score from BLOSUM matrix
                scores[c] = deviceBlosum[int(query_char) * BLOSUM_DIM + int(tc)];
                exp_scores[c] = expf(beta * scores[c]);
            }

            // ===== COMPUTE M AND F MATRICES (backward: dependencies from i+1, j+1) =====
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                int col = col_end_rev - c;

                // Get diagonal values [i+1, j+1] for ZM_bwd
                float ZM_diag, ZE_diag, ZF_diag;

                if (c == 0) {
                    // First (rightmost) column in thread - need from previous thread (to the right)
                    if (lane_id == 0) {
                        // Boundary condition: end of sequence
                        bool is_end = (row == query_len - 1 && col == target_len - 1);
                        ZM_diag = is_end ? 1.0f : 0.0f;
                        ZE_diag = 0.0f;
                        ZF_diag = 0.0f;
                    } else {
                        // Get first cell from previous thread's next row (lane_id - 1 has rightward columns)
                        ZM_diag = __shfl_up_sync(0xFFFFFFFF, ZM_next[K-1], 1);
                        ZE_diag = __shfl_up_sync(0xFFFFFFFF, ZE_next[K-1], 1);
                        ZF_diag = __shfl_up_sync(0xFFFFFFFF, ZF_next[K-1], 1);
                    }
                } else {
                    // Diagonal from same thread, next row
                    ZM_diag = ZM_next[c-1];
                    ZE_diag = ZE_next[c-1];
                    ZF_diag = ZF_next[c-1];
                }

                // Compute Z^M_bwd[i,j] = exp(beta * S) * (Z^M_bwd[i+1,j+1] + Z^E_bwd[i+1,j+1] + Z^F_bwd[i+1,j+1])
                // Note: In backward, we multiply by exp(score) at current cell
                ZM_curr[c] = exp_scores[c] * (ZM_diag + ZE_diag + ZF_diag);

                // Get bottom values [i+1, j] for ZF
                float ZM_bottom = ZM_next[c];
                float ZF_bottom = ZF_next[c];

                // Compute Z^F_bwd[i,j] = Z^M_bwd[i+1,j] * exp(gap_open) + Z^F_bwd[i+1,j] * exp(gap_extend)
                ZF_curr[c] = ZM_bottom * exp_go + ZF_bottom * exp_ge;
            }

            // ===== COMPUTE E MATRIX (sequential right-to-left) =====
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                int col = col_end_rev - c;

                // Get right values [i, j+1]
                float ZM_right, ZE_right;

                if (c == 0) {
                    // Rightmost column - need from previous thread
                    if (lane_id == 0) {
                        ZM_right = 0.0f;
                        ZE_right = 0.0f;
                    } else {
                        // Get from previous thread's last computed cell
                        ZM_right = __shfl_up_sync(0xFFFFFFFF, ZM_curr[K-1], 1);
                        ZE_right = __shfl_up_sync(0xFFFFFFFF, ZE_curr[K-1], 1);
                    }
                } else {
                    // Already computed in this thread
                    ZM_right = ZM_curr[c-1];
                    ZE_right = ZE_curr[c-1];
                }

                // Compute Z^E_bwd[i,j] = Z^M_bwd[i,j+1] * exp(gap_open) + Z^E_bwd[i,j+1] * exp(gap_extend)
                ZE_curr[c] = ZM_right * exp_go + ZE_right * exp_ge;
            }

            // ===== NORMALIZATION =====
            float max_val = 0.0f;
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                max_val = fmaxf(max_val, ZM_curr[c]);
                max_val = fmaxf(max_val, ZE_curr[c]);
                max_val = fmaxf(max_val, ZF_curr[c]);
            }

            // Warp reduction to find global max for this row
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
            }
            max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

            // Normalize all values
            if (max_val > 1e-30f) {
                const float inv_max = 1.0f / max_val;
                #pragma unroll
                for (int c = 0; c < num_cols; c++) {
                    ZM_curr[c] *= inv_max;
                    ZE_curr[c] *= inv_max;
                    ZF_curr[c] *= inv_max;
                }

                if (lane_id == 0) {
                    log_scales_bwd[warp_id * query_len + row] = logf(max_val);
                }
            } else {
                if (lane_id == 0) {
                    log_scales_bwd[warp_id * query_len + row] = -1e30f;
                }
            }

            // ===== WRITE TO GLOBAL MEMORY =====
            const size_t matrix_offset = size_t(warp_id) * query_len * target_len + size_t(row) * target_len;
            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                int col = col_end_rev - c;
                if (col >= 0) {
                    zm_matrix_bwd[matrix_offset + col] = logf(ZM_curr[c] + 1e-30f);
                }
            }

            // ===== UPDATE BUFFERS FOR NEXT ITERATION =====
            #pragma unroll
            for (int c = 0; c < K; c++) {
                ZM_next[c] = (c < num_cols) ? ZM_curr[c] : 0.0f;
                ZE_next[c] = (c < num_cols) ? ZE_curr[c] : 0.0f;
                ZF_next[c] = (c < num_cols) ? ZF_curr[c] : 0.0f;
            }

        } // if valid row

    } // for each iteration
}


// Posterior computation kernel - computes P[i,j] = fwd[i,j] * bwd[i,j] / Z
// Each warp processes one alignment
template<int WARP_SIZE, int K>
__global__ void posterior_kernel(
    const float* zm_fwd,           // log(ZM_fwd) values
    const float* zm_bwd,           // log(ZM_bwd) values
    const float* log_scales_fwd,
    const float* log_scales_bwd,
    const float* logZ,             // log partition function per alignment
    float* posteriors,             // Output: P[i,j]
    float* max_posteriors,         // Output: max P[i,j] per alignment
    const SequenceLengthT* query_lengths,
    const SequenceLengthT* target_lengths,
    const int num_alignments
)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_alignments) return;

    const int query_len = query_lengths[warp_id];
    const int target_len = target_lengths[warp_id];
    const float log_Z = logZ[warp_id];

    const size_t matrix_base = size_t(warp_id) * query_len * target_len;
    const size_t scale_base = size_t(warp_id) * query_len;

    float local_max = 0.0f;

    // Process matrix in row-major order, K columns per thread
    for (int row = 0; row < query_len; row++) {
        // Get scale factors for this row
        const float scale_fwd = log_scales_fwd[scale_base + row];
        const float scale_bwd = log_scales_bwd[scale_base + row];

        // Each lane processes K consecutive columns
        const int col_start = lane_id * K;

        #pragma unroll
        for (int c = 0; c < K; c++) {
            const int col = col_start + c;
            if (col < target_len) {
                const size_t idx = matrix_base + size_t(row) * target_len + col;

                // log P[i,j] = log_ZM_fwd[i,j] + scale_fwd[i] + log_ZM_bwd[i,j] + scale_bwd[i] - logZ
                const float log_fwd = zm_fwd[idx];
                const float log_bwd = zm_bwd[idx];
                const float log_posterior = log_fwd + scale_fwd + log_bwd + scale_bwd - log_Z;

                // Convert to probability
                const float posterior = expf(log_posterior);
                posteriors[idx] = posterior;

                local_max = fmaxf(local_max, posterior);
            }
        }
    }

    // Warp reduction for max posterior
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }

    if (lane_id == 0) {
        max_posteriors[warp_id] = local_max;
    }
}


// Kernel to compute log partition function from forward pass
// logZ = log(ZM_fwd[last_row, last_col]) + sum(log_scales_fwd)
__global__ void logZ_kernel(
    const float* zm_fwd,           // log(ZM_fwd) matrix
    const float* log_scales_fwd,
    float* logZ,                   // Output: log partition function per alignment
    const SequenceLengthT* query_lengths,
    const SequenceLengthT* target_lengths,
    const int num_alignments
)
{
    const int alignment_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (alignment_id >= num_alignments) return;

    const int query_len = query_lengths[alignment_id];
    const int target_len = target_lengths[alignment_id];

    // Get log(ZM_fwd) at the last cell
    const size_t matrix_base = size_t(alignment_id) * query_len * target_len;
    const size_t last_idx = matrix_base + size_t(query_len - 1) * target_len + (target_len - 1);
    float log_zm_end = zm_fwd[last_idx];

    // Sum all log scale factors
    const size_t scale_base = size_t(alignment_id) * query_len;
    float sum_log_scales = 0.0f;
    for (int i = 0; i < query_len; i++) {
        float scale = log_scales_fwd[scale_base + i];
        if (scale > -1e20f) {  // Skip invalid scales
            sum_log_scales += scale;
        }
    }

    // logZ = log(ZM_end) + sum(log_scales)
    logZ[alignment_id] = log_zm_end + sum_log_scales;
}


// Kernel to find max posterior per alignment (alternative to posterior_kernel when posteriors already computed)
__global__ void max_posterior_kernel(
    const float* posteriors,
    float* max_posteriors,
    const SequenceLengthT* query_lengths,
    const SequenceLengthT* target_lengths,
    const int num_alignments
)
{
    const int alignment_id = blockIdx.x;

    if (alignment_id >= num_alignments) return;

    const int query_len = query_lengths[alignment_id];
    const int target_len = target_lengths[alignment_id];
    const size_t matrix_size = size_t(query_len) * target_len;
    const size_t matrix_base = size_t(alignment_id) * matrix_size;

    // Parallel reduction with all threads in block
    extern __shared__ float sdata[];

    float local_max = 0.0f;
    for (size_t i = threadIdx.x; i < matrix_size; i += blockDim.x) {
        local_max = fmaxf(local_max, posteriors[matrix_base + i]);
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_posteriors[alignment_id] = sdata[0];
    }
}


} // namespace fwbw
} // namespace cudasw4

#endif

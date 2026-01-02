#include "fwbw_kernels.cuh"

namespace cudasw4{
namespace fwbw{

// Explicit template instantiations for common configurations
// WARP_SIZE = 32 (standard NVIDIA warp size)
// K = columns per thread (16, 20, 24 are good choices)
// BLOSUM_DIM = 20 or 25 depending on matrix

// Forward pass instantiations
template __global__ void forward_pass_kernel<32, 16, 20>(
    const char*, const char*, float*, float*, float*, float*,
    const size_t*, const size_t*, const SequenceLengthT*, const SequenceLengthT*,
    const float, const float, const float, const int
);

template __global__ void forward_pass_kernel<32, 20, 20>(
    const char*, const char*, float*, float*, float*, float*,
    const size_t*, const size_t*, const SequenceLengthT*, const SequenceLengthT*,
    const float, const float, const float, const int
);

template __global__ void forward_pass_kernel<32, 24, 20>(
    const char*, const char*, float*, float*, float*, float*,
    const size_t*, const size_t*, const SequenceLengthT*, const SequenceLengthT*,
    const float, const float, const float, const int
);

// Backward pass instantiations (updated signature with ze_row, zf_row)
template __global__ void backward_pass_kernel<32, 16, 20>(
    const char*, const char*, const float*, float*, float*, float*, float*,
    const size_t*, const size_t*, const SequenceLengthT*, const SequenceLengthT*,
    const float, const float, const float, const int
);

template __global__ void backward_pass_kernel<32, 20, 20>(
    const char*, const char*, const float*, float*, float*, float*, float*,
    const size_t*, const size_t*, const SequenceLengthT*, const SequenceLengthT*,
    const float, const float, const float, const int
);

template __global__ void backward_pass_kernel<32, 24, 20>(
    const char*, const char*, const float*, float*, float*, float*, float*,
    const size_t*, const size_t*, const SequenceLengthT*, const SequenceLengthT*,
    const float, const float, const float, const int
);

// Posterior kernel instantiations (updated signature with logZ and max_posteriors)
template __global__ void posterior_kernel<32, 16>(
    const float*, const float*, const float*, const float*, const float*,
    float*, float*, const SequenceLengthT*, const SequenceLengthT*, const int
);

template __global__ void posterior_kernel<32, 20>(
    const float*, const float*, const float*, const float*, const float*,
    float*, float*, const SequenceLengthT*, const SequenceLengthT*, const int
);

template __global__ void posterior_kernel<32, 24>(
    const float*, const float*, const float*, const float*, const float*,
    float*, float*, const SequenceLengthT*, const SequenceLengthT*, const int
);

// logZ kernel instantiation (non-templated)
// No explicit instantiation needed - it's not a template

// max_posterior kernel instantiation (non-templated)
// No explicit instantiation needed - it's not a template

} // namespace fwbw
} // namespace cudasw4


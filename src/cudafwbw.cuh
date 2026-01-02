#ifndef CUDAFWBW_CUH
#define CUDAFWBW_CUH

#include "hpc_helpers/cuda_raiiwrappers.cuh"
#include "hpc_helpers/all_helpers.cuh"
#include "hpc_helpers/simple_allocation.cuh"
#include "config.hpp"
#include "dbdata.hpp"
#include "types.hpp"
#include "fwbw_kernels.cuh"
#include "length_partitions.hpp"
#include "dbbatching.cuh"

#include <memory>
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace cudasw4{
namespace fwbw{

// Benchmark statistics for performance tracking
struct BenchmarkStats{
    double seconds = 0;
    double gcups = 0;
    int numOverflows = 0;
};

// Result structure for Forward-Backward alignment
struct ForwardBackwardResult{
    std::vector<float> logZ;                    // Log partition function per alignment
    std::vector<float> maxPosteriors;           // Max posterior probability per alignment
    std::vector<ReferenceIdT> referenceIds;     // Database sequence IDs
    BenchmarkStats stats;                       // Performance metrics

    // Optional: full posterior matrices (memory intensive!)
    std::vector<float> posteriors;              // Flattened posterior matrices
};

// Memory configuration
struct MemoryConfig{
    size_t maxBatchBytes = 128ull * 1024ull * 1024ull;       // 128 MB
    size_t maxBatchSequences = 10'000'000;
    size_t maxGpuMem = std::numeric_limits<size_t>::max();
};

// Main Forward-Backward class (parallel structure to CudaSW4)
class CudaFwBw{
public:
    template<class T>
    using MyPinnedBuffer = helpers::SimpleAllocationPinnedHost<T, 0>;
    template<class T>
    using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;

    // GPU Working Set (per-GPU memory and state)
    struct GpuWorkingSet{

        GpuWorkingSet(
            size_t gpumemlimit,
            size_t maxBatchBytes,
            size_t maxBatchSequences,
            const std::vector<DBdataView>& dbPartitions,
            const std::vector<DeviceBatchCopyToPinnedPlan>& dbBatches,
            bool computePosterior
        ){
            cudaGetDevice(&deviceId);

            size_t numSubjects = 0;
            size_t numSubjectBytes = 0;
            for(const auto& p : dbPartitions){
                numSubjects += p.numSequences();
                numSubjectBytes += p.numChars();
            }

            // Allocate query buffer
            d_query.resize(1024*1024);

            // Forward-Backward specific buffers
            maxBatchResultListSize = numSubjects;

            // Estimate max matrix size (conservative approach)
            constexpr size_t maxQueryLen = 1000;
            constexpr size_t maxTargetLen = 1000;

            // Per-alignment workspace - allocate for maximum expected sizes
            d_zm_matrix_fwd.resize(maxBatchResultListSize * maxQueryLen * maxTargetLen);
            d_zm_matrix_bwd.resize(maxBatchResultListSize * maxQueryLen * maxTargetLen);
            d_ze_row.resize(maxBatchResultListSize * maxTargetLen);
            d_zf_row.resize(maxBatchResultListSize * maxTargetLen);
            d_log_scales_fwd.resize(maxBatchResultListSize * maxQueryLen);
            d_log_scales_bwd.resize(maxBatchResultListSize * maxQueryLen);

            // Output: log partition functions and max posteriors
            d_logZ.resize(maxBatchResultListSize);
            d_maxPosteriors.resize(maxBatchResultListSize);

            if(computePosterior){
                d_posteriors.resize(maxBatchResultListSize * maxQueryLen * maxTargetLen);
            }

            // Database buffers
            numCopyBuffers = 2;
            h_chardata_vec.resize(numCopyBuffers);
            h_lengthdata_vec.resize(numCopyBuffers);
            h_offsetdata_vec.resize(numCopyBuffers);
            d_chardata_vec.resize(numCopyBuffers);
            d_lengthdata_vec.resize(numCopyBuffers);
            d_offsetdata_vec.resize(numCopyBuffers);

            // Metadata buffers
            d_query_offsets.resize(maxBatchResultListSize);
            d_target_offsets.resize(maxBatchResultListSize);
            d_query_lengths.resize(maxBatchResultListSize);
            d_target_lengths.resize(maxBatchResultListSize);

            // Allocate copy buffers
            for(int i = 0; i < numCopyBuffers; i++){
                h_chardata_vec[i].resize(maxBatchBytes);
                h_lengthdata_vec[i].resize(maxBatchSequences);
                h_offsetdata_vec[i].resize(maxBatchSequences+1);
                d_chardata_vec[i].resize(maxBatchBytes);
                d_lengthdata_vec[i].resize(maxBatchSequences);
                d_offsetdata_vec[i].resize(maxBatchSequences+1);
            }

            // Create CUDA streams
            copyStream = cudaStream_t{};
            cudaStreamCreate(&copyStream); CUERR;
            workStream = cudaStream_t{};
            cudaStreamCreate(&workStream); CUERR;

            // Check memory budget
            size_t usedGpuMem = 0;
            usedGpuMem += sizeof(float) * d_zm_matrix_fwd.size();
            usedGpuMem += sizeof(float) * d_zm_matrix_bwd.size();
            usedGpuMem += sizeof(float) * d_ze_row.size();
            usedGpuMem += sizeof(float) * d_zf_row.size();
            usedGpuMem += sizeof(float) * d_log_scales_fwd.size();
            usedGpuMem += sizeof(float) * d_log_scales_bwd.size();
            usedGpuMem += sizeof(float) * d_logZ.size();
            usedGpuMem += sizeof(float) * d_maxPosteriors.size();
            if(computePosterior){
                usedGpuMem += sizeof(float) * d_posteriors.size();
            }

            if(usedGpuMem > gpumemlimit){
                throw std::runtime_error("Out of memory for Forward-Backward working set");
            }
        }

        ~GpuWorkingSet(){
            if(copyStream){
                cudaStreamDestroy(copyStream);
            }
            if(workStream){
                cudaStreamDestroy(workStream);
            }
        }

        int deviceId;
        int numCopyBuffers;
        int maxBatchResultListSize;

        // CUDA streams
        cudaStream_t copyStream;
        cudaStream_t workStream;

        // Forward-Backward specific buffers
        MyDeviceBuffer<char> d_query;
        MyDeviceBuffer<float> d_zm_matrix_fwd;
        MyDeviceBuffer<float> d_zm_matrix_bwd;
        MyDeviceBuffer<float> d_ze_row;
        MyDeviceBuffer<float> d_zf_row;
        MyDeviceBuffer<float> d_log_scales_fwd;
        MyDeviceBuffer<float> d_log_scales_bwd;
        MyDeviceBuffer<float> d_logZ;
        MyDeviceBuffer<float> d_maxPosteriors;
        MyDeviceBuffer<float> d_posteriors;

        // Sequence metadata buffers
        MyDeviceBuffer<size_t> d_query_offsets;
        MyDeviceBuffer<size_t> d_target_offsets;
        MyDeviceBuffer<SequenceLengthT> d_query_lengths;
        MyDeviceBuffer<SequenceLengthT> d_target_lengths;

        // Database transfer buffers
        std::vector<MyPinnedBuffer<char>> h_chardata_vec;
        std::vector<MyPinnedBuffer<SequenceLengthT>> h_lengthdata_vec;
        std::vector<MyPinnedBuffer<size_t>> h_offsetdata_vec;
        std::vector<MyDeviceBuffer<char>> d_chardata_vec;
        std::vector<MyDeviceBuffer<SequenceLengthT>> d_lengthdata_vec;
        std::vector<MyDeviceBuffer<size_t>> d_offsetdata_vec;
    };

    // Constructor
    CudaFwBw(
        const std::vector<int>& deviceIds_,
        int results_per_query_,
        BlosumType blosumType_,
        const MemoryConfig& memoryConfig_,
        bool verbose_ = false
    ) : deviceIds(deviceIds_),
        results_per_query(results_per_query_),
        blosumType(blosumType_),
        memoryConfig(memoryConfig_),
        verbose(verbose_)
    {
        initializeGpus();
    }

    // Configuration methods
    void setGapOpenScore(int score){ gop = score; }
    void setGapExtendScore(int score){ gex = score; }

    void setDatabase(std::shared_ptr<DB> dbPtr){
        fullDB = dbPtr;
        dbIsReady = false;
    }

    void setBlosum(BlosumType blosumType_){ this->blosumType = blosumType_; }
    void setNumTop(int value){ results_per_query = value; }
    void setBeta(float value){ beta = value; }
    void setComputePosterior(bool value){ computePosterior = value; }

    // Database info methods
    size_t getReferenceLength(ReferenceIdT id) const{
        return fullDB.getData().getSequenceLength(id);
    }

    std::string getReferenceHeader(ReferenceIdT id) const{
        return fullDB.getData().getSequenceHeader(id);
    }

    // Main scanning method
    ForwardBackwardResult scan(const char* query, SequenceLengthT queryLength){
        if(!dbIsReady){
            makeReady();
        }

        // Set up query
        setQuery(query, queryLength);

        // Perform the scan
        scanDatabaseForQuery();

        // Collect and return results
        ForwardBackwardResult result;
        const int numGpus = deviceIds.size();

        // Collect results from all GPUs
        for(int gpu = 0; gpu < numGpus; gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;
            auto& ws = *workingSets[gpu];

            // Copy results back to host
            std::vector<float> h_logZ(ws.maxBatchResultListSize);
            std::vector<float> h_maxPost(ws.maxBatchResultListSize);

            cudaMemcpy(h_logZ.data(), ws.d_logZ.data(),
                      sizeof(float) * ws.maxBatchResultListSize,
                      cudaMemcpyDeviceToHost); CUERR;
            cudaMemcpy(h_maxPost.data(), ws.d_maxPosteriors.data(),
                      sizeof(float) * ws.maxBatchResultListSize,
                      cudaMemcpyDeviceToHost); CUERR;

            // Add to result
            result.logZ.insert(result.logZ.end(), h_logZ.begin(), h_logZ.end());
            result.maxPosteriors.insert(result.maxPosteriors.end(), h_maxPost.begin(), h_maxPost.end());
        }

        // Sort by logZ and take top-K
        std::vector<size_t> indices(result.logZ.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + std::min(size_t(results_per_query), indices.size()),
                         indices.end(),
                         [&](size_t a, size_t b){ return result.logZ[a] > result.logZ[b]; });

        // Keep only top-K
        ForwardBackwardResult topK;
        for(int i = 0; i < std::min(results_per_query, int(result.logZ.size())); i++){
            topK.logZ.push_back(result.logZ[indices[i]]);
            topK.maxPosteriors.push_back(result.maxPosteriors[indices[i]]);
            topK.referenceIds.push_back(indices[i]);
        }

        return topK;
    }

    // Initialization
    void makeReady(){
        if(verbose){
            std::cout << "Forward-Backward: Initializing database...\n";
        }

        const auto& dbData = fullDB.getData();
        const size_t numDBSequences = dbData.numSequences();
        maxBatchResultListSize = numDBSequences;

        // Compute length partitions
        computeTotalNumSequencePerLengthPartition();

        // Partition database among GPUs
        partitionDBAmongstGpus();

        // Create batching plans
        createDBBatchesForGpus();

        // Allocate GPU memory
        allocateGpuWorkingSets();

        dbIsReady = true;

        if(verbose){
            std::cout << "Forward-Backward: Database ready.\n";
        }
    }

private:
    // Initialize CUDA contexts and streams
    void initializeGpus(){
        const int numGpus = deviceIds.size();

        for(int i = 0; i < numGpus; i++){
            cudaSetDevice(deviceIds[i]); CUERR;
            helpers::init_cuda_context(); CUERR;
            cudaDeviceSetCacheConfig(cudaFuncCachePreferShared); CUERR;

            cudaMemPool_t mempool;
            cudaDeviceGetDefaultMemPool(&mempool, deviceIds[i]); CUERR;
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold); CUERR;

            gpuStreams.emplace_back();
            gpuEvents.emplace_back(cudaEventDisableTiming);
        }
    }

    // Compute number of sequences per length partition
    void computeTotalNumSequencePerLengthPartition(){
        auto lengthBoundaries = getLengthPartitionBoundaries();
        const int numLengthPartitions = lengthBoundaries.size();

        fullDB_numSequencesPerLengthPartition.resize(numLengthPartitions);

        const auto& dbData = fullDB.getData();
        auto partitionBegin = dbData.lengths();

        for(int i = 0; i < numLengthPartitions; i++){
            SequenceLengthT searchFor = lengthBoundaries[i];
            if(searchFor < std::numeric_limits<SequenceLengthT>::max()){
                searchFor += 1;
            }
            auto partitionEnd = std::lower_bound(
                partitionBegin,
                dbData.lengths() + dbData.numSequences(),
                searchFor
            );
            fullDB_numSequencesPerLengthPartition[i] = std::distance(partitionBegin, partitionEnd);
            partitionBegin = partitionEnd;
        }
    }

    // Partition database among GPUs
    void partitionDBAmongstGpus(){
        const int numGpus = deviceIds.size();
        const int numLengthPartitions = getLengthPartitionBoundaries().size();

        subPartitionsForGpus.clear();
        lengthPartitionIdsForGpus.clear();
        numSequencesPerGpu.clear();

        const auto& data = fullDB.getData();

        subPartitionsForGpus.resize(numGpus);
        lengthPartitionIdsForGpus.resize(numGpus);
        numSequencesPerGpu.resize(numGpus, 0);

        std::vector<size_t> numSequencesPerLengthPartitionPrefixSum(numLengthPartitions, 0);
        for(int i = 0; i < numLengthPartitions-1; i++){
            numSequencesPerLengthPartitionPrefixSum[i+1] =
                numSequencesPerLengthPartitionPrefixSum[i] + fullDB_numSequencesPerLengthPartition[i];
        }

        std::vector<DBdataView> dbPartitionsByLengthPartitioning;
        for(int i = 0; i < numLengthPartitions; i++){
            size_t begin = numSequencesPerLengthPartitionPrefixSum[i];
            size_t end = begin + fullDB_numSequencesPerLengthPartition[i];
            dbPartitionsByLengthPartitioning.emplace_back(data, begin, end);
        }

        // Distribute length partitions across GPUs
        for(int lengthPartitionId = 0; lengthPartitionId < numLengthPartitions; lengthPartitionId++){
            const auto& lengthPartition = dbPartitionsByLengthPartitioning[lengthPartitionId];
            const auto partitionedByGpu = partitionDBdata_by_numberOfChars(
                lengthPartition,
                lengthPartition.numChars() / numGpus
            );

            for(int gpu = 0; gpu < numGpus; gpu++){
                if(gpu < int(partitionedByGpu.size())){
                    subPartitionsForGpus[gpu].push_back(partitionedByGpu[gpu]);
                    lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
                }else{
                    subPartitionsForGpus[gpu].push_back(DBdataView(data, 0, 0));
                    lengthPartitionIdsForGpus[gpu].push_back(0);
                }
            }
        }

        for(int i = 0; i < numGpus; i++){
            for(const auto& p : subPartitionsForGpus[i]){
                numSequencesPerGpu[i] += p.numSequences();
            }
        }
    }

    // Create database batching plans
    void createDBBatchesForGpus(){
        const int numGpus = deviceIds.size();

        batchPlans.clear();
        batchPlans.resize(numGpus);

        for(int gpu = 0; gpu < numGpus; gpu++){
            batchPlans[gpu] = computeDbCopyPlan(
                subPartitionsForGpus[gpu],
                memoryConfig.maxBatchBytes,
                memoryConfig.maxBatchSequences
            );
        }
    }

    // Allocate GPU working sets
    void allocateGpuWorkingSets(){
        const int numGpus = deviceIds.size();
        workingSets.clear();
        workingSets.resize(numGpus);

        if(verbose){
            std::cout << "Allocating Forward-Backward GPU memory...\n";
        }

        for(int gpu = 0; gpu < numGpus; gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;

            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem); CUERR;
            constexpr size_t safety = 256*1024*1024;
            size_t memlimit = std::min(freeMem, memoryConfig.maxGpuMem);
            if(memlimit > safety){
                memlimit -= safety;
            }

            if(verbose){
                std::cout << "GPU " << gpu << " can use " << memlimit << " bytes.\n";
            }

            workingSets[gpu] = std::make_unique<GpuWorkingSet>(
                memlimit,
                memoryConfig.maxBatchBytes,
                memoryConfig.maxBatchSequences,
                subPartitionsForGpus[gpu],
                batchPlans[gpu],
                computePosterior
            );
        }
    }

    // Set query sequence
    void setQuery(const char* query, SequenceLengthT queryLength){
        currentQueryLength = queryLength;
        currentQueryLengthWithPadding = queryLength; // Simplified, no padding needed for FwBw

        const int numGpus = deviceIds.size();

        for(int gpu = 0; gpu < numGpus; gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;
            auto& ws = *workingSets[gpu];

            // Copy query to device
            cudaMemcpy(ws.d_query.data(), query, queryLength, cudaMemcpyHostToDevice); CUERR;
        }
    }

    // Main database scanning orchestration
    void scanDatabaseForQuery(){
        processQueryOnGpus();
    }

    // Main computation: process query on all GPUs
    void processQueryOnGpus(){
        const int numGpus = deviceIds.size();

        for(int gpu = 0; gpu < numGpus; gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;
            auto& ws = *workingSets[gpu];

            // Process each batch
            const auto& batches = batchPlans[gpu];
            size_t globalOffset = 0;

            for(const auto& batch : batches){
                // Upload database batch
                uploadBatch(gpu, batch, globalOffset);

                // Launch forward pass
                launchForwardPass(gpu, batch.usedSeq, globalOffset);

                // Launch backward pass
                launchBackwardPass(gpu, batch.usedSeq, globalOffset);

                // Compute log partition functions
                computeLogPartitionFunctions(gpu, batch.usedSeq, globalOffset);

                // Compute posterior probabilities (if enabled)
                launchPosteriorComputation(gpu, batch.usedSeq, globalOffset);

                globalOffset += batch.usedSeq;
            }

            cudaDeviceSynchronize(); CUERR;
        }
    }

    // Upload a batch of database sequences
    void uploadBatch(int gpu, const DeviceBatchCopyToPinnedPlan& batch, size_t globalOffset){
        auto& ws = *workingSets[gpu];

        // Copy database sequences to device
        const auto& view = subPartitionsForGpus[gpu][batch.partitionId];

        cudaMemcpy(ws.d_chardata_vec[0].data(),
                  view.chars() + batch.offsetsOffset,
                  batch.usedBytes,
                  cudaMemcpyHostToDevice); CUERR;

        cudaMemcpy(ws.d_lengthdata_vec[0].data(),
                  view.lengths() + batch.lengthsOffset,
                  batch.usedSeq * sizeof(SequenceLengthT),
                  cudaMemcpyHostToDevice); CUERR;

        cudaMemcpy(ws.d_offsetdata_vec[0].data(),
                  view.offsets() + batch.offsetsOffset,
                  (batch.usedSeq + 1) * sizeof(size_t),
                  cudaMemcpyHostToDevice); CUERR;
    }

    // Launch forward pass kernel
    void launchForwardPass(int gpu, int numSeq, size_t globalOffset){
        auto& ws = *workingSets[gpu];

        // Kernel configuration
        constexpr int WARP_SIZE = 32;
        constexpr int K = 20;  // Columns per thread
        constexpr int BLOSUM_DIM = 20;

        const int numWarps = numSeq;
        const int numThreads = numWarps * WARP_SIZE;
        const int numBlocks = (numThreads + 255) / 256;

        // Launch kernel
        forward_pass_kernel<WARP_SIZE, K, BLOSUM_DIM><<<numBlocks, 256, 0, ws.workStream>>>(
            ws.d_query.data(),
            ws.d_chardata_vec[0].data(),
            ws.d_zm_matrix_fwd.data() + globalOffset * currentQueryLength * 1000,  // Simplified offset
            ws.d_ze_row.data() + globalOffset * 1000,
            ws.d_zf_row.data() + globalOffset * 1000,
            ws.d_log_scales_fwd.data() + globalOffset * currentQueryLength,
            ws.d_query_offsets.data(),
            ws.d_offsetdata_vec[0].data(),
            ws.d_query_lengths.data(),
            ws.d_lengthdata_vec[0].data(),
            beta,
            float(gop),
            float(gex),
            numSeq
        );
        CUERR;
    }

    // Launch backward pass kernel
    void launchBackwardPass(int gpu, int numSeq, size_t globalOffset){
        auto& ws = *workingSets[gpu];

        // Kernel configuration - same as forward pass
        constexpr int WARP_SIZE = 32;
        constexpr int K = 20;  // Columns per thread
        constexpr int BLOSUM_DIM = 20;

        const int numWarps = numSeq;
        const int numThreads = numWarps * WARP_SIZE;
        const int numBlocks = (numThreads + 255) / 256;

        // Launch backward pass kernel
        backward_pass_kernel<WARP_SIZE, K, BLOSUM_DIM><<<numBlocks, 256, 0, ws.workStream>>>(
            ws.d_query.data(),
            ws.d_chardata_vec[0].data(),
            ws.d_zm_matrix_fwd.data() + globalOffset * currentQueryLength * 1000,  // Forward matrix (for reference)
            ws.d_zm_matrix_bwd.data() + globalOffset * currentQueryLength * 1000,  // Output backward matrix
            ws.d_ze_row.data() + globalOffset * 1000,  // Workspace
            ws.d_zf_row.data() + globalOffset * 1000,  // Workspace
            ws.d_log_scales_bwd.data() + globalOffset * currentQueryLength,
            ws.d_query_offsets.data(),
            ws.d_offsetdata_vec[0].data(),
            ws.d_query_lengths.data(),
            ws.d_lengthdata_vec[0].data(),
            beta,
            float(gop),
            float(gex),
            numSeq
        );
        CUERR;
    }

    // Compute log partition functions from forward pass results
    void computeLogPartitionFunctions(int gpu, int numSeq, size_t globalOffset){
        auto& ws = *workingSets[gpu];

        // Launch logZ computation kernel
        const int numBlocks = (numSeq + 255) / 256;

        logZ_kernel<<<numBlocks, 256, 0, ws.workStream>>>(
            ws.d_zm_matrix_fwd.data() + globalOffset * currentQueryLength * 1000,
            ws.d_log_scales_fwd.data() + globalOffset * currentQueryLength,
            ws.d_logZ.data() + globalOffset,
            ws.d_query_lengths.data(),
            ws.d_lengthdata_vec[0].data(),
            numSeq
        );
        CUERR;
    }

    // Launch posterior computation kernel
    void launchPosteriorComputation(int gpu, int numSeq, size_t globalOffset){
        if (!computePosterior) return;

        auto& ws = *workingSets[gpu];

        constexpr int WARP_SIZE = 32;
        constexpr int K = 20;

        const int numWarps = numSeq;
        const int numThreads = numWarps * WARP_SIZE;
        const int numBlocks = (numThreads + 255) / 256;

        posterior_kernel<WARP_SIZE, K><<<numBlocks, 256, 0, ws.workStream>>>(
            ws.d_zm_matrix_fwd.data() + globalOffset * currentQueryLength * 1000,
            ws.d_zm_matrix_bwd.data() + globalOffset * currentQueryLength * 1000,
            ws.d_log_scales_fwd.data() + globalOffset * currentQueryLength,
            ws.d_log_scales_bwd.data() + globalOffset * currentQueryLength,
            ws.d_logZ.data() + globalOffset,
            ws.d_posteriors.data() + globalOffset * currentQueryLength * 1000,
            ws.d_maxPosteriors.data() + globalOffset,
            ws.d_query_lengths.data(),
            ws.d_lengthdata_vec[0].data(),
            numSeq
        );
        CUERR;
    }

    // Member variables
    bool verbose = false;
    bool dbIsReady = false;
    bool computePosterior = true;

    int gop = -11;
    int gex = -1;
    int results_per_query = 10;
    int maxBatchResultListSize = 0;
    float beta = 1.0f;

    BlosumType blosumType = BlosumType::BLOSUM62_20;
    MemoryConfig memoryConfig;

    std::vector<int> deviceIds;
    std::vector<std::unique_ptr<GpuWorkingSet>> workingSets;
    std::vector<cudaStream_t> gpuStreams;
    std::vector<helpers::CudaEvent> gpuEvents;

    std::shared_ptr<DB> fullDB;

    // Database partitioning data
    std::vector<size_t> fullDB_numSequencesPerLengthPartition;
    std::vector<std::vector<DBdataView>> subPartitionsForGpus;
    std::vector<std::vector<int>> lengthPartitionIdsForGpus;
    std::vector<size_t> numSequencesPerGpu;
    std::vector<std::vector<DeviceBatchCopyToPinnedPlan>> batchPlans;

    SequenceLengthT currentQueryLength = 0;
    SequenceLengthT currentQueryLengthWithPadding = 0;
};

} // namespace fwbw
} // namespace cudasw4

#endif

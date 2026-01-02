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

            // Linear Memory Workspace
            // We only need buffers proportional to target length for the current row
            d_prev_row_buffer.resize(maxBatchResultListSize * maxTargetLen);
            d_curr_row_buffer.resize(maxBatchResultListSize * maxTargetLen);
            d_log_scales.resize(maxBatchResultListSize * maxQueryLen);

            // Output: log partition functions and max posteriors
            d_logZ.resize(maxBatchResultListSize);
            d_maxPosteriors.resize(maxBatchResultListSize);

            // Optional: for future full matrix support, we can use a smaller pinned buffer
            // but for now we focus on linear memory scan.
            
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

            // Check memory budget (much lower now)
            size_t usedGpuMem = 0;
            usedGpuMem += sizeof(float) * d_prev_row_buffer.size();
            usedGpuMem += sizeof(float) * d_curr_row_buffer.size();
            usedGpuMem += sizeof(float) * d_log_scales.size();
            usedGpuMem += sizeof(float) * d_logZ.size();
            usedGpuMem += sizeof(float) * d_maxPosteriors.size();

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
        // Forward-Backward specific buffers (Linear Memory)
        MyDeviceBuffer<char> d_query;
        MyDeviceBuffer<float> d_prev_row_buffer; // Stores M for previous row
        MyDeviceBuffer<float> d_curr_row_buffer; // Stores M for current row
        MyDeviceBuffer<float> d_log_scales;
        MyDeviceBuffer<float> d_logZ;
        MyDeviceBuffer<float> d_maxPosteriors;

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
        fullDB = AnyDBWrapper(dbPtr);
        dbIsReady = false;
    }

    void setDatabase(std::shared_ptr<DBWithVectors> dbPtr){
        fullDB = AnyDBWrapper(dbPtr);
        dbIsReady = false;
    }

    void setBlosum(BlosumType blosumType_){ this->blosumType = blosumType_; }
    void setNumTop(int value){ results_per_query = value; }
    void setBeta(float value){ beta = value; }
    void setComputePosterior(bool value){ computePosterior = value; }

    // Database info methods
    size_t getReferenceLength(ReferenceIdT id) const{
        const auto& data = fullDB.getData();
        return data.lengths()[id];
    }

    std::string getReferenceHeader(ReferenceIdT id) const{
        const auto& data = fullDB.getData();
        const char* const headerBegin = data.headers() + data.headerOffsets()[id];
        const char* const headerEnd = data.headers() + data.headerOffsets()[id+1];
        return std::string(headerBegin, std::distance(headerBegin, headerEnd));
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

    // Compute database copy plan for batching
    std::vector<DeviceBatchCopyToPinnedPlan> computeDbCopyPlan(
        const std::vector<DBdataView>& dbPartitions,
        size_t MAX_CHARDATA_BYTES,
        size_t MAX_SEQ
    ) const {
        std::vector<DeviceBatchCopyToPinnedPlan> result;
    
        size_t currentCopyPartition = 0;
        size_t currentCopySeqInPartition = 0;
    
        while(currentCopyPartition < dbPartitions.size()){
            
            size_t usedBytes = 0;
            size_t usedSeq = 0;
    
            DeviceBatchCopyToPinnedPlan plan;
    
            while(currentCopyPartition < dbPartitions.size()){
                if(dbPartitions[currentCopyPartition].numSequences() == 0){
                    currentCopyPartition++;
                    continue;
                }
    
                size_t remainingBytes = MAX_CHARDATA_BYTES - usedBytes;
                
                auto dboffsetsBegin = dbPartitions[currentCopyPartition].offsets() + currentCopySeqInPartition;
                auto dboffsetsEnd = dbPartitions[currentCopyPartition].offsets() + dbPartitions[currentCopyPartition].numSequences() + 1;
                
                auto searchFor = dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition] + remainingBytes + 1;
                auto it = std::lower_bound(dboffsetsBegin, dboffsetsEnd, searchFor);
    
                size_t numToCopyByBytes = 0;
                if(it != dboffsetsBegin){
                    numToCopyByBytes = std::distance(dboffsetsBegin, it) - 1;
                }
                if(numToCopyByBytes == 0 && currentCopySeqInPartition == 0){
                    break;
                }
                
                size_t remainingSeq = MAX_SEQ - usedSeq;            
                size_t numToCopyBySeq = std::min(dbPartitions[currentCopyPartition].numSequences() - currentCopySeqInPartition, remainingSeq);
                size_t numToCopy = std::min(numToCopyByBytes, numToCopyBySeq);
    
                if(numToCopy > 0){
                    DeviceBatchCopyToPinnedPlan::CopyRange copyRange;
                    copyRange.lengthPartitionId = currentCopyPartition;
                    copyRange.currentCopyPartition = currentCopyPartition;
                    copyRange.currentCopySeqInPartition = currentCopySeqInPartition;
                    copyRange.numToCopy = numToCopy;
                    plan.copyRanges.push_back(copyRange);
    
                    if(usedSeq == 0){
                        plan.h_partitionIds.push_back(currentCopyPartition);
                        plan.h_numPerPartition.push_back(numToCopy);
                    }else{
                        if(plan.h_partitionIds.back() == int(currentCopyPartition)){
                            plan.h_numPerPartition.back() += numToCopy;
                        }else{
                            plan.h_partitionIds.push_back(currentCopyPartition);
                            plan.h_numPerPartition.push_back(numToCopy);
                        }
                    }
                    usedBytes += (dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition+numToCopy] 
                        - dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition]);
                    usedSeq += numToCopy;
    
                    currentCopySeqInPartition += numToCopy;
                    if(currentCopySeqInPartition == dbPartitions[currentCopyPartition].numSequences()){
                        currentCopySeqInPartition = 0;
                        currentCopyPartition++;
                    }
                }else{
                    break;
                }
            }
    
            plan.usedBytes = usedBytes;
            plan.usedSeq = usedSeq;    
            
            if(usedSeq == 0 && currentCopyPartition < dbPartitions.size() && dbPartitions[currentCopyPartition].numSequences() > 0){
                break;
            }
    
            if(plan.usedSeq > 0){
                result.push_back(plan);
            }
        }
    
        return result;
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
    // STUB IMPLEMENTATION - fills output with placeholder values
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

                // Launch forward pass (Linear Memory - Computes logZ)
                launchForwardPass(gpu, batch.usedSeq, globalOffset);

                // Backward pass and Posterior computation are disabled for Phase 1 (Scanning)
                // They require O(MN) memory or checkpointing, which we skip to prevent OOM
                /*
                launchBackwardPass(gpu, batch.usedSeq, globalOffset);
                computeLogPartitionFunctions(gpu, batch.usedSeq, globalOffset);
                launchPosteriorComputation(gpu, batch.usedSeq, globalOffset);
                */

                globalOffset += batch.usedSeq;
            }

            cudaDeviceSynchronize(); CUERR;
        }
    }

    // Upload a batch of database sequences
    void uploadBatch(int gpu, const DeviceBatchCopyToPinnedPlan& batch, size_t /*globalOffset*/){
        auto& ws = *workingSets[gpu];

        // Use the existing batch copy utility function
        executeCopyPlanH2DDirect(
            batch,
            ws.d_chardata_vec[0].data(),
            ws.d_lengthdata_vec[0].data(),
            ws.d_offsetdata_vec[0].data(),
            subPartitionsForGpus[gpu],
            ws.workStream
        );
    }

    // Launch forward pass kernel (Linear Memory)
    void launchForwardPass(int gpu, int numSeq, size_t globalOffset){
        auto& ws = *workingSets[gpu];

        // Kernel configuration
        constexpr int WARP_SIZE = 32;
        constexpr int K = 20;  // Columns per thread
        constexpr int BLOSUM_DIM = 20;

        const int numWarps = numSeq;
        const int numThreads = numWarps * WARP_SIZE;
        const int numBlocks = (numThreads + 255) / 256;

        // Calculate workspace offset for this batch within the GPU buffer
        // Note: ws.d_prev_row_buffer is size `maxBatchResultListSize * maxTargetLen`
        // We need to point to the correct start for `numSeq` sequences.
        // Assuming `globalOffset` is the sequence index in the current batch relative to batch start?
        // No, globalOffset is global DB data offset. 
        // Here we need offset into the GPU workspace buffer `d_prev_row_buffer`.
        // Since we reuse the workspace for each query, and we process batches sequentially on GPU,
        // we can just use 0 offset if we reset every batch? 
        // Wait, GpuWorkingSet is large enough for `maxBatchResultListSize` (total seqs on GPU).
        // So we should use `globalOffset` here if `globalOffset` tracks sequences processed so far on this GPU.
        // `scanDatabaseForQuery` loop tracks `globalOffset` relative to partition start.
        // Partition start corresponds to index 0 in `ws` buffers?
        // Yes, `d_query_offsets` etc are resized to `maxBatchResultListSize`.
        
        // Offset in floats (assuming maxTargetLen = 1000 for allocation purposes)
        // We should really use a computed offset array if lengths vary greatly to save memory,
        // but for now we follow the simple Strided allocation from GpuWorkingSet constructor.
        size_t row_buffer_offset = globalOffset * 1000; // Hardcoded 1000 stride matching allocation

        // Launch linear kernel
        forward_linear_kernel<WARP_SIZE, K, BLOSUM_DIM><<<numBlocks, 256, 0, ws.workStream>>>(
            ws.d_query.data(),
            ws.d_chardata_vec[0].data(),
            ws.d_prev_row_buffer.data() + row_buffer_offset,
            ws.d_curr_row_buffer.data() + row_buffer_offset,
            ws.d_log_scales.data() + globalOffset * currentQueryLength,
            ws.d_logZ.data() + globalOffset, // Output logZ
            ws.d_query_offsets.data() + globalOffset,
            ws.d_offsetdata_vec[0].data(), // DB offsets are absolute handled by kernel
            ws.d_query_lengths.data() + globalOffset,
            ws.d_lengthdata_vec[0].data(), // DB lengths are absolute
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
    std::vector<CudaEvent> gpuEvents;

    AnyDBWrapper fullDB;

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

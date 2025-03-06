#pragma once

#include "LegionUtilities.hpp"
#include "TaskIDs.hpp"

#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <legion.h>

#include "CUDAUtilities.hpp"


#define THREADS_PER_BLOCK 128

namespace LegionSolvers {


#ifdef __CUDACC__
__device__ inline size_t global_tid_1d() {
    return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}
#endif

inline size_t get_num_blocks_1d(size_t threads) {
    return (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

class InitNCCLUniqueIDTask {
public:
    static const int TASK_ID =
        LEGION_SOLVERS_TASK_ID_ORIGIN + INIT_NCCL_UNIQUE_ID_META_TASK_ID;
    static constexpr const char *task_name = "load_cuda_libs";
    using return_type = ncclUniqueId;
    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
    static void preregister(bool verbose);
};


// LoadCUDALibsTask is a task that loads the CUDA libraries
// on each GPU in the system.
class LoadCUDALibsTask {
public:
    static const int TASK_ID =
        LEGION_SOLVERS_TASK_ID_ORIGIN + LOAD_CUDA_LIBS_META_TASK_ID;
    static constexpr const char *task_name = "load_cuda_libs";
    using return_type = void;
    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
    static void preregister(bool verbose);
};

// Function to actually perform initialization of CUDA modules
// on each GPU. This should only be called once the Legion
// runtime has started.
void loadCUDALibs(Legion::Context ctx, Legion::Runtime *rt);

} // namespace LegionSolvers

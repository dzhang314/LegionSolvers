#pragma once

#include "LegionUtilities.hpp"
#include "TaskIDs.hpp"

#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <legion.h>

#define CHECK_CUDA(expr)                                                       \
    do {                                                                       \
        cudaError_t __result__ = (expr);                                       \
        LegionSolvers::check_cuda(__result__, __FILE__, __LINE__);             \
    } while (false)

#define CHECK_CUDA_STREAM(stream)                                              \
    do {                                                                       \
        CHECK_CUDA(cudaStreamSynchronize(stream));                             \
        CHECK_CUDA(cudaPeekAtLastError());                                     \
    } while (false)

#define CHECK_CUBLAS(expr)                                                     \
    do {                                                                       \
        cublasStatus_t __result__ = (expr);                                    \
        LegionSolvers::check_cublas(__result__, __FILE__, __LINE__);           \
    } while (false)

#define CHECK_CUSPARSE(expr)                                                   \
    do {                                                                       \
        cusparseStatus_t result = (expr);                                      \
        LegionSolvers::check_cusparse(result, __FILE__, __LINE__);              \
    } while (false)

namespace LegionSolvers {

__host__ inline void check_cuda(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        fprintf(
            stderr,
            "CUDA failure with error %s (%s) in file %s at line %d\n",
            cudaGetErrorString(error),
            cudaGetErrorName(error),
            file,
            line
        );
        assert(false);
    }
}

__host__ inline void check_cublas(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "cuBLAS failure with error code %d in file %s at line %d\n",
            status,
            file,
            line);
    assert(false);
  }
}


__host__ inline void check_cusparse(cusparseStatus_t status, const char *file, int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(
            stderr,
            "CUSPARSE failure with error code %d (%s) in file %s at line %d\n",
            status,
            cusparseGetErrorString(status),
            file,
            line
        );
        assert(false);
    }
}

// StreamView is a managed view of a CUDA stream. This code is
// inspired from Legate's CUDA StreamView.
struct StreamView {
  public:
    StreamView(cudaStream_t stream) : valid_(true), stream_(stream) {}
    ~StreamView();

  public:
    StreamView(const StreamView &) = delete;
    StreamView &operator=(const StreamView &) = delete;

  public:
    StreamView(StreamView &&);
    StreamView &operator=(StreamView &&);

  public:
    operator cudaStream_t() const { return stream_; }

  private:
    bool valid_;
    cudaStream_t stream_;
};

// Return a cached stream for the current GPU.
StreamView get_cached_stream();
// Method to get the cuSPARSE handle associated with the executing GPU.
cusparseHandle_t get_cusparse();
// Method to get the cuBLAS handle associated with the executing GPU.
cublasHandle_t get_cublas();

// CUDALibraries is a struct that manages handles on libraries
// like cuSPARSE and (in the future) cuBLAS, as well as what
// stream should kernels execute on.
struct CUDALibraries {
  public:
    CUDALibraries();

  private:
    // Prevent copying and overwriting.
    CUDALibraries(const CUDALibraries &rhs) = delete;
    CUDALibraries &operator=(const CUDALibraries &rhs) = delete;

  public:
    cublasHandle_t get_cublas();
    cusparseHandle_t get_cusparse();
    cudaStream_t get_stream();

  private:
    cublasHandle_t cublas_;
    cusparseHandle_t cusparse_;
    cudaStream_t stream_;
};

// LoadCUDALibsTask is a task that loads the CUDA libraries
// on each GPU in the system.
class LoadCUDALibsTask {
  public:
    static const int TASK_ID =
        LEGION_SOLVERS_TASK_ID_ORIGIN + LOAD_CUDALIBS_TASK_ID;
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

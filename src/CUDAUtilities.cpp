#include "CUDAUtilities.hpp"

#include <cassert> // for assert
#include <cstdio>  // for std::fprintf, stderr

#include <legion.h> // for Legion::*

using LegionSolvers::CUDALibraryContext;
using LegionSolvers::CUDAStreamView;


void LegionSolvers::check_cuda(cudaError_t status, const char *file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA failure with status code %d (%s: %s) "
            "in file %s at line %d\n",
            status,
            cudaGetErrorName(status),
            cudaGetErrorString(status),
            file,
            line
        );
        assert(false);
    }
}


void LegionSolvers::check_cublas(
    cublasStatus_t status, const char *file, int line
) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "cuBLAS failure with status code %d "
            "in file %s at line %d\n",
            status,
            file,
            line
        );
        assert(false);
    }
}


void LegionSolvers::check_cusparse(
    cusparseStatus_t status, const char *file, int line
) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "cuSPARSE failure with status code %d (%s: %s) "
            "in file %s at line %d\n",
            status,
            cusparseGetErrorName(status),
            cusparseGetErrorString(status),
            file,
            line
        );
        assert(false);
    }
}


cudaStream_t CUDALibraryContext::get_cuda_stream() {
    if (cuda_stream == nullptr) {
        CHECK_CUDA(
            cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking)
        );
    }
    return cuda_stream;
}


cublasHandle_t CUDALibraryContext::get_cublas_handle() {
    if (cublas_handle == nullptr) {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
    }
    return cublas_handle;
}


cusparseHandle_t CUDALibraryContext::get_cusparse_handle() {
    if (cusparse_handle == nullptr) {
        CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
    }
    return cusparse_handle;
}


constexpr CUDAStreamView::CUDAStreamView(CUDAStreamView &&v) noexcept
    : cuda_stream(v.cuda_stream)
    , valid(v.valid) {
    v.valid = false;
}


constexpr CUDAStreamView &CUDAStreamView::operator=(CUDAStreamView &&v
) noexcept {
    cuda_stream = v.cuda_stream;
    valid = v.valid;
    v.valid = false;
    return *this;
}


CUDAStreamView::~CUDAStreamView() {
    // TODO (rohany): static variable to control whether we synchronize
    //                at the end of each task
    if (valid) {
#ifndef NDEBUG
        CHECK_CUDA_STREAM(cuda_stream);
// LegionSolvers does not currently use the Realm CUDA hijack,
// so we can let Realm handle synchronization of the stream.
// #else
// CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
#endif
    }
}


static CUDALibraryContext &get_cuda_library_context() {
    static CUDALibraryContext CUDA_LIBRARY_CONTEXTS[LEGION_MAX_NUM_PROCS];
    const Legion::Processor proc = Legion::Processor::get_executing_processor();
    assert(proc.kind() == Legion::Processor::TOC_PROC);
    return CUDA_LIBRARY_CONTEXTS[proc.id & (LEGION_MAX_NUM_PROCS - 1)];
}


CUDAStreamView LegionSolvers::get_cuda_stream() {
    return CUDAStreamView(get_cuda_library_context().get_cuda_stream());
}


cublasHandle_t LegionSolvers::get_cublas_handle() {
    return get_cuda_library_context().get_cublas_handle();
}


cusparseHandle_t LegionSolvers::get_cusparse_handle() {
    return get_cuda_library_context().get_cusparse_handle();
}

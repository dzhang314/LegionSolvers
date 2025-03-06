#ifndef LEGION_SOLVERS_CUDA_UTILITIES_HPP_INCLUDED
#define LEGION_SOLVERS_CUDA_UTILITIES_HPP_INCLUDED

#include <cassert> // for assert

#include <cublas_v2.h>    // for cublas*
#include <cuda_runtime.h> // for cuda*
#include <cusparse.h>     // for cusparse*
#include <nccl.h>

namespace LegionSolvers {


void check_cuda(cudaError_t status, const char *file, int line);

void check_cublas(cublasStatus_t status, const char *file, int line);

void check_cusparse(cusparseStatus_t status, const char *file, int line);

void check_nccl(ncclResult_t result, const char* file, int line);

#define CHECK_CUDA(expr)                                                       \
    do {                                                                       \
        cudaError_t status_ = (expr);                                          \
        LegionSolvers::check_cuda(status_, __FILE__, __LINE__);                \
    } while (false)

#define CHECK_CUDA_STREAM(stream)                                              \
    do {                                                                       \
        CHECK_CUDA(cudaStreamSynchronize(stream));                             \
        CHECK_CUDA(cudaPeekAtLastError());                                     \
    } while (false)

#define CHECK_CUBLAS(expr)                                                     \
    do {                                                                       \
        cublasStatus_t status_ = (expr);                                       \
        LegionSolvers::check_cublas(status_, __FILE__, __LINE__);              \
    } while (false)

#define CHECK_CUSPARSE(expr)                                                   \
    do {                                                                       \
        cusparseStatus_t status_ = (expr);                                     \
        LegionSolvers::check_cusparse(status_, __FILE__, __LINE__);            \
    } while (false)

#define CHECK_NCCL(...)               \
  do {                                       \
    const ncclResult_t status_ = __VA_ARGS__; \
    LegionSolvers::check_nccl(status_, __FILE__, __LINE__);  \
  } while (false)

class CUDALibraryContext {

    cudaStream_t cuda_stream;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    ncclComm_t nccl_comm;

public:

    constexpr CUDALibraryContext() noexcept
        : cuda_stream(nullptr)
        , cublas_handle(nullptr)
        , cusparse_handle(nullptr)
        , nccl_comm(nullptr) {}

    CUDALibraryContext(const CUDALibraryContext &) = delete;
    CUDALibraryContext(CUDALibraryContext &&) = delete;
    CUDALibraryContext &operator=(const CUDALibraryContext &) = delete;
    CUDALibraryContext &operator=(CUDALibraryContext &&) = delete;

    cudaStream_t get_cuda_stream();
    cublasHandle_t get_cublas_handle();
    cusparseHandle_t get_cusparse_handle();
    ncclComm_t get_nccl_comm();
    void set_nccl_comm(ncclComm_t);

}; // class CUDALibraryContext


class CUDAStreamView {

    cudaStream_t cuda_stream;
    bool valid;

public:

    constexpr CUDAStreamView(cudaStream_t stream) noexcept
        : cuda_stream(stream)
        , valid(true) {}

    CUDAStreamView(const CUDAStreamView &) = delete;
    CUDAStreamView &operator=(const CUDAStreamView &) = delete;

    constexpr CUDAStreamView(CUDAStreamView &&) noexcept;
    constexpr CUDAStreamView &operator=(CUDAStreamView &&) noexcept;

    ~CUDAStreamView();

    operator cudaStream_t() const noexcept { return cuda_stream; }

}; // class CUDAStreamView


CUDAStreamView get_cuda_stream();

cusparseHandle_t get_cusparse_handle();

cublasHandle_t get_cublas_handle();

ncclComm_t get_nccl_comm();
void set_nccl_comm(ncclComm_t);


// clang-format off
template <typename ENTRY_T> constexpr cudaDataType CUDA_DATA_TYPE = static_cast<cudaDataType>(-1);
template <> constexpr cudaDataType CUDA_DATA_TYPE<float> = CUDA_R_32F;
template <> constexpr cudaDataType CUDA_DATA_TYPE<double> = CUDA_R_64F;

template <typename COORD_T> constexpr cusparseIndexType_t CUSPARSE_INDEX_TYPE = static_cast<cusparseIndexType_t>(-1);
template <> constexpr cusparseIndexType_t CUSPARSE_INDEX_TYPE<int> = CUSPARSE_INDEX_32I;
// NOTE: strangely, there is no CUSPARSE_INDEX_32U
template <> constexpr cusparseIndexType_t CUSPARSE_INDEX_TYPE<long long> = CUSPARSE_INDEX_64I;

template <typename ENTRY_T> inline void cublas_axpy(cublasHandle_t handle, int n, const ENTRY_T *alpha, const ENTRY_T *x, int incx, ENTRY_T *y, int incy) { assert(false); }
template <> inline void cublas_axpy<float>(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) { CHECK_CUBLAS(cublasSaxpy(handle, n, alpha, x, incx, y, incy)); }
template <> inline void cublas_axpy<double>(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) { CHECK_CUBLAS(cublasDaxpy(handle, n, alpha, x, incx, y, incy)); }

template <typename ENTRY_T> inline void cublas_dot(cublasHandle_t handle, int n, const ENTRY_T *x, int incx, const ENTRY_T *y, int incy, ENTRY_T *result) { assert(false); }
template <> inline void cublas_dot<float>(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) { CHECK_CUBLAS(cublasSdot(handle, n, x, incx, y, incy, result)); }
template <> inline void cublas_dot<double>(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) { CHECK_CUBLAS(cublasDdot(handle, n, x, incx, y, incy, result)); }

template <typename ENTRY_T> inline void cublas_scal(cublasHandle_t handle, int n, const ENTRY_T *alpha, ENTRY_T *x, int incx) { assert(false); }
template <> inline void cublas_scal<float>(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) { CHECK_CUBLAS(cublasSscal(handle, n, alpha, x, incx)); }
template <> inline void cublas_scal<double>(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) { CHECK_CUBLAS(cublasDscal(handle, n, alpha, x, incx)); }
// clang-format on


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_CUDA_UTILITIES_HPP_INCLUDED

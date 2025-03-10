#include "LinearAlgebraTasks.hpp"

#include <cassert> // for assert
#include <cmath>   // for std::fma

#include "CudaLibs.hpp"
#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*
#include "Pitches.hpp"

using namespace LegionSolvers;


template <typename ENTRY_T, int DIM, typename COORD_T>
void ScalTask<ENTRY_T, DIM, COORD_T>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS) {

    // Grab our stream and cuBLAS handle.
    auto stream = get_cuda_stream();
    auto handle = get_cublas_handle();
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    assert(regions.size() == 1);
    const auto &x = regions[0];

    assert(task->regions.size() == 1);
    const auto &x_req = task->regions[0];

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> x_reader_writer(x, x_fid);

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    assert(x_domain.dense());

    // If there are no points to process, exit.
    if (x_domain.empty()) return;

    // TODO (rohany): I'm not sure about what the right value for incx and
    //  incy are. It depends on what layouts we're getting the input
    //  vectors in. If we're getting exact layouts than this should be fine.
    //  If we're getting some subslice of a larger region then this probably
    //  won't work.
    // Finally make the cuBLAS call.
    cublas_scal<ENTRY_T>(
        handle,
        x_domain.get_volume(),
        &alpha,
        x_reader_writer.ptr(x_domain.lo()),
        1
    );
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void AxpyTask<ENTRY_T, DIM, COORD_T>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS) {

    // Grab our stream and cuBLAS handle.
    auto stream = get_cuda_stream();
    auto handle = get_cublas_handle();
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    assert(regions.size() == 2);
    const auto &y = regions[0];
    const auto &x = regions[1];

    assert(task->regions.size() == 2);
    const auto &y_req = task->regions[0];
    const auto &x_req = task->regions[1];

    assert(y_req.privilege_fields.size() == 1);
    const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y_reader_writer{y, y_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> x_reader{x, x_fid};

    const Legion::Domain y_domain =
        rt->get_index_space_domain(ctx, y_req.region.get_index_space());

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    assert(y_domain == x_domain);
    assert(y_domain.dense());

    // If there are no points to process, exit.
    if (y_domain.empty()) return;

    // TODO (rohany): I'm not sure about what the right value for incx and
    //  incy are. It depends on what layouts we're getting the input
    //  vectors in. If we're getting exact layouts than this should be fine.
    //  If we're getting some subslice of a larger region then this probably
    //  won't work.
    // Finally make the cuBLAS call.
    cublas_axpy<ENTRY_T>(
        handle,
        y_domain.get_volume(),
        &alpha,
        x_reader.ptr(x_domain.lo()),
        1,
        y_reader_writer.ptr(y_domain.lo()),
        1
    );
}


// cuBLAS doesn't have a XPAY kernel for some reason, so just
// write it out by hand.
template <typename ENTRY_T, int DIM, typename COORD_T>
__global__ void xpay_kernel(
    size_t volume,
    ENTRY_T alpha,
    Pitches<DIM - 1, COORD_T> pitches,
    const Legion::Point<DIM, COORD_T> lo,
    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y,
    AffineReader<ENTRY_T, DIM, COORD_T> x
) {
    const auto idx = global_tid_1d();
    if (idx >= volume) return;
    auto point = pitches.unflatten(idx, lo);
    y[point] = std::fma(alpha, y[point], x[point]);
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void XpayTask<ENTRY_T, DIM, COORD_T>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS) {

    assert(regions.size() == 2);
    const auto &y = regions[0];
    const auto &x = regions[1];

    assert(task->regions.size() == 2);
    const auto &y_req = task->regions[0];
    const auto &x_req = task->regions[1];

    assert(y_req.privilege_fields.size() == 1);
    const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y_reader_writer{y, y_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> x_reader{x, x_fid};

    const Legion::Domain y_domain =
        rt->get_index_space_domain(ctx, y_req.region.get_index_space());

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    assert(y_domain == x_domain);
    assert(y_domain.dense());

    // If there are no points to process, exit.
    if (y_domain.empty()) return;

    auto stream = get_cuda_stream();
    Pitches<DIM - 1, COORD_T> pitches;
    auto volume = pitches.flatten(y_domain.bounds<DIM, COORD_T>());
    auto blocks = get_num_blocks_1d(volume);
    xpay_kernel<ENTRY_T, DIM, COORD_T>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            volume, alpha, pitches, y_domain.lo(), y_reader_writer, x_reader
        );
}


template <typename ENTRY_T, int DIM, typename COORD_T>
ENTRY_T DotTask<ENTRY_T, DIM, COORD_T>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS
) {
    // Grab our stream and cuBLAS handle.
    auto stream = get_cuda_stream();
    auto handle = get_cublas_handle();
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    assert(regions.size() == 2);
    const auto &v = regions[0];
    const auto &w = regions[1];

    assert(task->regions.size() == 2);
    const auto &v_req = task->regions[0];
    const auto &w_req = task->regions[1];

    assert(v_req.privilege_fields.size() == 1);
    const Legion::FieldID v_fid = *v_req.privilege_fields.begin();

    assert(w_req.privilege_fields.size() == 1);
    const Legion::FieldID w_fid = *w_req.privilege_fields.begin();

    AffineReader<ENTRY_T, DIM, COORD_T> v_reader{v, v_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> w_reader{w, w_fid};

    const Legion::Domain v_domain =
        rt->get_index_space_domain(ctx, v_req.region.get_index_space());

    const Legion::Domain w_domain =
        rt->get_index_space_domain(ctx, w_req.region.get_index_space());

    assert(v_domain == w_domain);
    assert(v_domain.dense());

    ENTRY_T result = static_cast<ENTRY_T>(0);
    if (v_domain.empty()) { return result; }

    // TODO (rohany): I'm not sure about what the right value for incx and
    //  incy are. It depends on what layouts we're getting the input
    //  vectors in. If we're getting exact layouts than this should be fine.
    //  If we're getting some subslice of a larger region then this probably
    //  won't work.
    // Finally make the cuBLAS call.
    cublas_dot<ENTRY_T>(
        handle,
        v_domain.get_volume(),
        v_reader.ptr(v_domain.lo()),
        1,
        w_reader.ptr(w_domain.lo()),
        1,
        &result
    );

    // It appears that cublas might be synchronizing the stream for us
    // since it sees result is a host pointer, but it's clearer to do
    // this directly in the task itself.
    CHECK_CUDA(cudaStreamSynchronize(stream));

    return result;
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F64
// clang-format on

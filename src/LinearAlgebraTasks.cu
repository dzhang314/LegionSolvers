#include "LinearAlgebraTasks.hpp"

#include <cassert> // for assert
#include <cmath>   // for std::fma

#include "CudaLibs.hpp"
#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*
#include "Pitches.hpp"

using namespace LegionSolvers;

template <typename ENTRY_T>
__global__ void fill1(ENTRY_T* ptr) {
  if (threadIdx.x == 0) {
    *ptr = static_cast<ENTRY_T>(1);
  }
}
template <typename ENTRY_T>
__global__ void alpha_kern(const ENTRY_T* f0, const ENTRY_T* f1, ENTRY_T* out) {
  if (threadIdx.x == 0) {
    *out = *f0 / *f1;
  }
}
template <typename ENTRY_T>
__global__ void alpha_kern(const ENTRY_T* f0, const ENTRY_T* f1, const ENTRY_T* f2, ENTRY_T* out) {
  if (threadIdx.x == 0) {
    *out = (*f0 * *f1) / *f2;
  }
}
template <typename ENTRY_T>
__global__ void alpha_kern(const ENTRY_T* f0, const ENTRY_T* f1, const ENTRY_T* f2, const ENTRY_T* f3, ENTRY_T* out) {
  if (threadIdx.x == 0) {
    *out = (*f0 * *f1) / (*f2 * *f3);
  }
}

template <typename ENTRY_T>
const ENTRY_T* get_alpha_gpu(const int32_t* args, const std::vector<Legion::Future>& futures, cudaStream_t stream) {
    if (futures.size() == 0) {
	Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::GPU_FB_MEM, nullptr, 16);
	// Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::Z_COPY_MEM, nullptr, 16);
        fill1<ENTRY_T><<<1, 32, 0, stream>>>(res.ptr(0));
	return res.ptr(0);
    } else if (futures.size() == 1) {
	return (const ENTRY_T*)futures[0].get_buffer(Legion::Memory::GPU_FB_MEM);
	// return (const ENTRY_T*)futures[0].get_buffer(Legion::Memory::Z_COPY_MEM);
    } else if (futures.size() == 2) {
	Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::GPU_FB_MEM, nullptr, 16);
	// Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::Z_COPY_MEM, nullptr, 16);
        // const ENTRY_T* f0 = (const ENTRY_T*)futures[0].get_buffer(Legion::Memory::Z_COPY_MEM);
        // const ENTRY_T* f1 = (const ENTRY_T*)futures[1].get_buffer(Legion::Memory::Z_COPY_MEM);
        const ENTRY_T* f0 = (const ENTRY_T*)futures[args[0]].get_buffer(Legion::Memory::GPU_FB_MEM);
        const ENTRY_T* f1 = (const ENTRY_T*)futures[args[1]].get_buffer(Legion::Memory::GPU_FB_MEM);
	alpha_kern<ENTRY_T><<<1, 32, 0, stream>>>(f0, f1, res.ptr(0));
	return res.ptr(0);
    } else if (futures.size() == 3) {
	Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::GPU_FB_MEM, nullptr, 16);
	// Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::Z_COPY_MEM, nullptr, 16);
        // const ENTRY_T* f0 = (const ENTRY_T*)futures[0].get_buffer(Legion::Memory::Z_COPY_MEM);
        // const ENTRY_T* f1 = (const ENTRY_T*)futures[1].get_buffer(Legion::Memory::Z_COPY_MEM);
        // const ENTRY_T* f2 = (const ENTRY_T*)futures[2].get_buffer(Legion::Memory::Z_COPY_MEM);
        const ENTRY_T* f0 = (const ENTRY_T*)futures[args[0]].get_buffer(Legion::Memory::GPU_FB_MEM);
        const ENTRY_T* f1 = (const ENTRY_T*)futures[args[1]].get_buffer(Legion::Memory::GPU_FB_MEM);
        const ENTRY_T* f2 = (const ENTRY_T*)futures[args[2]].get_buffer(Legion::Memory::GPU_FB_MEM);
	alpha_kern<ENTRY_T><<<1, 32, 0, stream>>>(f0, f1, f2, res.ptr(0));
	return res.ptr(0);
    } else if (futures.size() == 4) {
	Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::GPU_FB_MEM, nullptr, 16);
	// Legion::DeferredBuffer<ENTRY_T, 1> res(Legion::Rect<1>(0, 0), Legion::Memory::Z_COPY_MEM, nullptr, 16);
        // const ENTRY_T* f0 = (const ENTRY_T*)futures[0].get_buffer(Legion::Memory::Z_COPY_MEM);
        // const ENTRY_T* f1 = (const ENTRY_T*)futures[1].get_buffer(Legion::Memory::Z_COPY_MEM);
        // const ENTRY_T* f2 = (const ENTRY_T*)futures[2].get_buffer(Legion::Memory::Z_COPY_MEM);
        // const ENTRY_T* f3 = (const ENTRY_T*)futures[3].get_buffer(Legion::Memory::Z_COPY_MEM);
        const ENTRY_T* f0 = (const ENTRY_T*)futures[args[0]].get_buffer(Legion::Memory::GPU_FB_MEM);
        const ENTRY_T* f1 = (const ENTRY_T*)futures[args[1]].get_buffer(Legion::Memory::GPU_FB_MEM);
        const ENTRY_T* f2 = (const ENTRY_T*)futures[args[2]].get_buffer(Legion::Memory::GPU_FB_MEM);
        const ENTRY_T* f3 = (const ENTRY_T*)futures[args[3]].get_buffer(Legion::Memory::GPU_FB_MEM);
	alpha_kern<ENTRY_T><<<1, 32, 0, stream>>>(f0, f1, f2, f3, res.ptr(0));
	return res.ptr(0);
    } else {
        assert(false);
	return nullptr;
    }
}

template <typename ENTRY_T, int DIM, typename COORD_T>
__global__ void scal_kernel(
    size_t volume,
    const ENTRY_T* alpha,
    Pitches<DIM - 1, COORD_T> pitches,
    const Legion::Point<DIM, COORD_T> lo,
    AffineReaderWriter<ENTRY_T, DIM, COORD_T> x
) {
    const auto idx = global_tid_1d();
    if (idx >= volume) return;
    auto point = pitches.unflatten(idx, lo);
    x[point] = *alpha * x[point];
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void ScalTask<ENTRY_T, DIM, COORD_T>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS) {

    // Grab our stream and cuBLAS handle.
    auto stream = get_cuda_stream();
    // auto handle = get_cublas_handle();
    // CHECK_CUBLAS(cublasSetStream(handle, stream));

    assert(regions.size() == 1);
    const auto &x = regions[0];

    assert(task->regions.size() == 1);
    const auto &x_req = task->regions[0];

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> x_reader_writer(x, x_fid);

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    assert(x_domain.dense());

    // If there are no points to process, exit.
    if (x_domain.empty()) return;

    const ENTRY_T* alpha = get_alpha_gpu<ENTRY_T>((const int32_t*)task->args, task->futures, stream);

    // TODO (rohany): I'm not sure about what the right value for incx and
    //  incy are. It depends on what layouts we're getting the input
    //  vectors in. If we're getting exact layouts than this should be fine.
    //  If we're getting some subslice of a larger region then this probably
    //  won't work.
    // Finally make the cuBLAS call.
    // cublas_scal<ENTRY_T>(
    //     handle,
    //     x_domain.get_volume(),
    //     alpha,
    //     x_reader_writer.ptr(x_domain.lo()),
    //     1
    // );

    Pitches<DIM - 1, COORD_T> pitches;
    auto volume = pitches.flatten(x_domain.bounds<DIM, COORD_T>());
    auto blocks = get_num_blocks_1d(volume);
    scal_kernel<ENTRY_T, DIM, COORD_T>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            volume, alpha, pitches, x_domain.lo(), x_reader_writer
        );
}

template <typename ENTRY_T, int DIM, typename COORD_T>
__global__ void axpy_kernel(
    size_t volume,
    const ENTRY_T* alpha,
    Pitches<DIM - 1, COORD_T> pitches,
    const Legion::Point<DIM, COORD_T> lo,
    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y,
    AffineReader<ENTRY_T, DIM, COORD_T> x
) {
    const auto idx = global_tid_1d();
    if (idx >= volume) return;
    auto point = pitches.unflatten(idx, lo);
    y[point] = std::fma(*alpha, x[point], y[point]);
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void AxpyTask<ENTRY_T, DIM, COORD_T>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS) {

    // Grab our stream and cuBLAS handle.
    auto stream = get_cuda_stream();
    // auto handle = get_cublas_handle();
    // CHECK_CUBLAS(cublasSetStream(handle, stream));

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

    const ENTRY_T* alpha = get_alpha_gpu<ENTRY_T>((const int32_t*)task->args, task->futures, stream);

    Pitches<DIM - 1, COORD_T> pitches;
    auto volume = pitches.flatten(y_domain.bounds<DIM, COORD_T>());
    auto blocks = get_num_blocks_1d(volume);
    axpy_kernel<ENTRY_T, DIM, COORD_T>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            volume, alpha, pitches, y_domain.lo(), y_reader_writer, x_reader
        );

    // It's unbelievable, but cublas axpy does not respect stream ordering
    // for the alpha variable!!!!
    // TODO (rohany): I'm not sure about what the right value for incx and
    //  incy are. It depends on what layouts we're getting the input
    //  vectors in. If we're getting exact layouts than this should be fine.
    //  If we're getting some subslice of a larger region then this probably
    //  won't work.
    // Finally make the cuBLAS call.
    // cublas_axpy<ENTRY_T>(
    //     handle,
    //     y_domain.get_volume(),
    //     alpha,
    //     x_reader.ptr(x_domain.lo()),
    //     1,
    //     y_reader_writer.ptr(y_domain.lo()),
    //     1
    // );
}


// cuBLAS doesn't have a XPAY kernel for some reason, so just
// write it out by hand.
template <typename ENTRY_T, int DIM, typename COORD_T>
__global__ void xpay_kernel(
    size_t volume,
    const ENTRY_T* alpha,
    Pitches<DIM - 1, COORD_T> pitches,
    const Legion::Point<DIM, COORD_T> lo,
    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y,
    AffineReader<ENTRY_T, DIM, COORD_T> x
) {
    const auto idx = global_tid_1d();
    if (idx >= volume) return;
    auto point = pitches.unflatten(idx, lo);
    y[point] = std::fma(*alpha, y[point], x[point]);
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

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y_reader_writer{y, y_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> x_reader{x, x_fid};

    const Legion::Domain y_domain =
        rt->get_index_space_domain(ctx, y_req.region.get_index_space());

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    assert(y_domain == x_domain);
    assert(y_domain.dense());

    auto stream = get_cuda_stream();

    // If there are no points to process, exit.
    if (y_domain.empty()) return;

    const ENTRY_T* alpha = get_alpha_gpu<ENTRY_T>((const int32_t*)task->args, task->futures, stream);
    Pitches<DIM - 1, COORD_T> pitches;
    auto volume = pitches.flatten(y_domain.bounds<DIM, COORD_T>());
    auto blocks = get_num_blocks_1d(volume);
    xpay_kernel<ENTRY_T, DIM, COORD_T>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            volume, alpha, pitches, y_domain.lo(), y_reader_writer, x_reader
        );
}

// Kernel for dot product with shared memory and warp reductions
template<typename ENTRY_T>
__global__ void dotProductKernel(const ENTRY_T *d_A, const ENTRY_T *d_B, ENTRY_T *d_result, size_t n) {
    __shared__ ENTRY_T s_data[THREADS_PER_BLOCK];  // Shared memory for partial results

    int64_t idx = global_tid_1d();
    int64_t threadId = threadIdx.x;

    ENTRY_T partial_sum = 0.0f;

    // Each thread computes a part of the dot product
    if (idx < n) {
        partial_sum = d_A[idx] * d_B[idx];
    }

    // Store the partial sum in shared memory
    s_data[threadId] = partial_sum;

    __syncthreads();  // Synchronize threads in the block to ensure data is ready

    // Warp-level reduction within the block
    if (threadId < 64) s_data[threadId] += s_data[threadId + 64];
    __syncthreads();
    if (threadId < 32) s_data[threadId] += s_data[threadId + 32];
    __syncthreads();
    ENTRY_T local_result = s_data[threadId];
    for (int offset = 16; offset > 0; offset /= 2) {
      local_result += __shfl_down_sync(0xFFFFFFFF, local_result, offset);
    }

    // The first thread of each block writes the result to global memory
    if (threadId == 0) {
        atomicAdd(d_result, local_result);
    }
}

template <typename ENTRY_T, int DIM, typename COORD_T>
void DotTask<ENTRY_T, DIM, COORD_T>::cuda_task(const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p) {
    const Legion::Task *task; Legion::Context ctx; Legion::Runtime *rt;
    const std::vector<Legion::PhysicalRegion> *regionsp;
    Legion::Runtime::legion_task_preamble(args, arglen, p, task, regionsp, ctx, rt);
    const std::vector<Legion::PhysicalRegion>& regions = *regionsp;

    // Grab our stream and cuBLAS handle.
    auto stream = get_cuda_stream();
    // auto handle = get_cublas_handle();
    // CHECK_CUBLAS(cublasSetStream(handle, stream));

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

    // Allocate the result.
    Legion::UntypedDeferredValue resultu(sizeof(ENTRY_T), Legion::Memory::GPU_FB_MEM);
    Legion::DeferredValue<ENTRY_T> result = resultu;
    CHECK_CUDA(cudaMemsetAsync(result.ptr(), 0, sizeof(ENTRY_T), stream));

    // TODO (rohany): I'm not sure about what the right value for incx and
    //  incy are. It depends on what layouts we're getting the input
    //  vectors in. If we're getting exact layouts than this should be fine.
    //  If we're getting some subslice of a larger region then this probably
    //  won't work.
    // Finally make the cuBLAS call.
    // cublas_dot<ENTRY_T>(
    //     handle,
    //     v_domain.get_volume(),
    //     v_reader.ptr(v_domain.lo()),
    //     1,
    //     w_reader.ptr(w_domain.lo()),
    //     1,
    //     result.ptr()
    // );

    if (!v_domain.empty()) {
      int64_t blockSize = THREADS_PER_BLOCK;
      int64_t numBlocks = (v_domain.get_volume() + blockSize - 1) / blockSize;
      dotProductKernel<ENTRY_T><<<numBlocks, blockSize, 0, stream>>>(v_reader.ptr(v_domain.lo()), w_reader.ptr(w_domain.lo()), result.ptr(), v_domain.get_volume());
    }

    // Also do NCCL.
    if (task->index_domain.get_volume() > 1) {
      auto comm = get_nccl_comm();
      if (sizeof(ENTRY_T) == 4) {
        CHECK_NCCL(ncclAllReduce(result.ptr(), result.ptr(), 1, ncclFloat32, ncclSum, comm, stream));
      } else {
        CHECK_NCCL(ncclAllReduce(result.ptr(), result.ptr(), 1, ncclFloat64, ncclSum, comm, stream));
      }
    }

    result.finalize(ctx);
}

// template <typename ENTRY_T, int DIM, typename COORD_T>
// ENTRY_T DotTask<ENTRY_T, DIM, COORD_T>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS
// ) {
//     // Grab our stream and cuBLAS handle.
//     auto stream = get_cuda_stream();
//     auto handle = get_cublas_handle();
//     CHECK_CUBLAS(cublasSetStream(handle, stream));
// 
//     assert(regions.size() == 2);
//     const auto &v = regions[0];
//     const auto &w = regions[1];
// 
//     assert(task->regions.size() == 2);
//     const auto &v_req = task->regions[0];
//     const auto &w_req = task->regions[1];
// 
//     assert(v_req.privilege_fields.size() == 1);
//     const Legion::FieldID v_fid = *v_req.privilege_fields.begin();
// 
//     assert(w_req.privilege_fields.size() == 1);
//     const Legion::FieldID w_fid = *w_req.privilege_fields.begin();
// 
//     AffineReader<ENTRY_T, DIM, COORD_T> v_reader{v, v_fid};
//     AffineReader<ENTRY_T, DIM, COORD_T> w_reader{w, w_fid};
// 
//     const Legion::Domain v_domain =
//         rt->get_index_space_domain(ctx, v_req.region.get_index_space());
// 
//     const Legion::Domain w_domain =
//         rt->get_index_space_domain(ctx, w_req.region.get_index_space());
// 
//     assert(v_domain == w_domain);
//     assert(v_domain.dense());
// 
//     ENTRY_T result = static_cast<ENTRY_T>(0);
//     if (v_domain.empty()) { return result; }
// 
//     // TODO (rohany): I'm not sure about what the right value for incx and
//     //  incy are. It depends on what layouts we're getting the input
//     //  vectors in. If we're getting exact layouts than this should be fine.
//     //  If we're getting some subslice of a larger region then this probably
//     //  won't work.
//     // Finally make the cuBLAS call.
//     cublas_dot<ENTRY_T>(
//         handle,
//         v_domain.get_volume(),
//         v_reader.ptr(v_domain.lo()),
//         1,
//         w_reader.ptr(w_domain.lo()),
//         1,
//         &result
//     );
// 
//     // It appears that cublas might be synchronizing the stream for us
//     // since it sees result is a host pointer, but it's clearer to do
//     // this directly in the task itself.
//     CHECK_CUDA(cudaStreamSynchronize(stream));
// 
//     return result;
// }

#define REALM_TASK_ARGS const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor proc

// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 1, int>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 2, int>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 3, int>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 1, unsigned>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 2, unsigned>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 3, unsigned>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 1, long long>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 2, long long>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<float, 3, long long>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 1, int>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 2, int>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, int>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 3, int>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 1, unsigned>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 2, unsigned>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, unsigned>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 3, unsigned>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 1, long long>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 2, long long>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
            template void DotTask<double, 3, long long>::cuda_task(REALM_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F64
// clang-format on

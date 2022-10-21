#include "LinearAlgebraTasks.hpp"

#include "CuBLASHelpers.hpp"
#include "CudaLibs.hpp"
#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*
#include "Pitches.hpp"

#include <cmath>   // for std::fma

using namespace LegionSolvers;

template <typename ENTRY_T, int DIM, typename COORD_T>
void ScalTask<ENTRY_T, DIM, COORD_T>::gpu_task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
  assert(false);
}

template <typename ENTRY_T, int DIM, typename COORD_T>
void AxpyTask<ENTRY_T, DIM, COORD_T>::gpu_task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
  // Grab our stream and cuBLAS handle.
  auto stream = get_cached_stream();
  auto handle = get_cublas();
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
  cublasAXPY<ENTRY_T>(
      handle,
      y_domain.get_volume(),
      &alpha,
      x_reader.ptr(x_domain.lo()),
      sizeof(ENTRY_T),
      y_reader_writer.ptr(y_domain.lo()),
      sizeof(ENTRY_T)
  );
}

// cuBLAS doesn't have a XPAY kernel for some reason, so just
// write it out by hand.
template <typename ENTRY_T, int DIM, typename COORD_T>
__global__
void xpay_kernel(size_t volume,
                 ENTRY_T alpha,
                 Pitches<DIM - 1, COORD_T> pitches,
                 const Legion::Point<DIM, COORD_T> lo,
                 AffineReaderWriter<ENTRY_T, DIM, COORD_T> y,
                 AffineReader<ENTRY_T, DIM, COORD_T> x) {
  const auto idx = global_tid_1d();
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, lo);
  y[point] = std::fma(alpha, y[point], x[point]);
}

template <typename ENTRY_T, int DIM, typename COORD_T>
void XpayTask<ENTRY_T, DIM, COORD_T>::gpu_task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
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

  auto stream = get_cached_stream();
  Pitches<DIM - 1, COORD_T> pitches;
  auto volume = pitches.flatten(y_domain.bounds<DIM, COORD_T>());
  auto blocks = get_num_blocks_1d(volume);
  xpay_kernel<ENTRY_T, DIM, COORD_T><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    volume,
    alpha,
    pitches,
    y_domain.lo(),
    y_reader_writer,
    x_reader
  );
}

template <typename ENTRY_T, int DIM, typename COORD_T>
ENTRY_T DotTask<ENTRY_T, DIM, COORD_T>::gpu_task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
  assert(false);
  return ENTRY_T{0};
}

// clang-format off
#ifdef LEGION_SOLVERS_USE_FLOAT
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<float, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<float, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template float DotTask<float, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_FLOAT
#ifdef LEGION_SOLVERS_USE_DOUBLE
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 1, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 2, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 3, int>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 1, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 2, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 3, unsigned>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 1, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 2, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void AxpyTask<double, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template void XpayTask<double, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
            template double DotTask<double, 3, long long>::gpu_task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_DOUBLE
// clang-format on

#include "LinearAlgebraTasks.hpp"

#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*

using LegionSolvers::AxpyTask;
using LegionSolvers::DotTask;
using LegionSolvers::ScalTask;
using LegionSolvers::XpayTask;

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
  assert(false);
}

template <typename ENTRY_T, int DIM, typename COORD_T>
void XpayTask<ENTRY_T, DIM, COORD_T>::gpu_task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
  assert(false);
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

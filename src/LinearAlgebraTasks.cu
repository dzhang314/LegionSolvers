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

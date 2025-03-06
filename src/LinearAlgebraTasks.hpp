#ifndef LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED

#include "LegionUtilities.hpp" // for TaskFlags
#include "TaskBaseClasses.hpp" // for TaskTDI
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


template <typename ENTRY_T, int DIM, typename COORD_T>
struct ScalTask
    : public TaskTDI<SCAL_TASK_BLOCK_ID, ScalTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "scal";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


template <typename ENTRY_T, int DIM, typename COORD_T>
struct AxpyTask
    : public TaskTDI<AXPY_TASK_BLOCK_ID, AxpyTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "axpy";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


template <typename ENTRY_T, int DIM, typename COORD_T>
struct XpayTask
    : public TaskTDI<XPAY_TASK_BLOCK_ID, XpayTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "xpay";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


template <typename ENTRY_T, int DIM, typename COORD_T>
struct DotTask
    : public TaskTDI<DOT_TASK_BLOCK_ID, DotTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "dot_product";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(ENTRY_T);
#ifdef LEGION_USE_CUDA
    static void preregister_fb_future_dot(bool verbose) {
      if (verbose) {
          std::cout << "[LegionSolvers] Registering task " << TaskTDI<DOT_TASK_BLOCK_ID, DotTask, ENTRY_T, DIM, COORD_T>::task_name()
                    << " with ID " << TaskTDI<DOT_TASK_BLOCK_ID, DotTask, ENTRY_T, DIM, COORD_T>::task_id << "." << std::endl;
      }
      Legion::TaskVariantRegistrar registrar{TaskTDI<DOT_TASK_BLOCK_ID, DotTask, ENTRY_T, DIM, COORD_T>::task_id, TaskTDI<DOT_TASK_BLOCK_ID, DotTask, ENTRY_T, DIM, COORD_T>::task_name().c_str()};
      registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::TOC_PROC));
      registrar.set_leaf(flags & TaskFlags::LEAF);
      registrar.set_inner(flags & TaskFlags::INNER);
      registrar.set_idempotent(flags & TaskFlags::IDEMPOTENT);
      registrar.set_replicable(flags & TaskFlags::REPLICABLE);
      Legion::Runtime::preregister_task_variant(registrar, Legion::CodeDescriptor(cuda_task));
    }
    static void cuda_task(const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor proc);
#endif

    // LEGION_SOLVERS_DECLARE_CUDA_TASK;
    
};


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED

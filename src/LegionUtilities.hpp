#ifndef LEGION_SOLVERS_LEGION_UTILITIES_HPP_INCLUDED
#define LEGION_SOLVERS_LEGION_UTILITIES_HPP_INCLUDED

#include <cstddef>     // for std::size_t
#include <cstdint>     // for std::uint8_t
#include <string>      // for std::string
#include <type_traits> // for std::is_void_v
#include <vector>      // for std::vector

#include <legion.h> // for Legion::*

#include "LibraryOptions.hpp" // for LEGION_SOLVERS_CHECK_BOUNDS

// clang-format off

#define LEGION_SOLVERS_TASK_ARGS \
    const Legion::Task *task, \
    const std::vector<Legion::PhysicalRegion> &regions, \
    Legion::Context ctx, Legion::Runtime *rt

#define LEGION_SOLVERS_DECLARE_TASK(RETURN_TYPE) \
    using return_type = RETURN_TYPE; \
    static return_type task_body(LEGION_SOLVERS_TASK_ARGS)

#define LEGION_SOLVERS_DECLARE_CUDA_TASK \
    static return_type cuda_task_body(LEGION_SOLVERS_TASK_ARGS)

#define LEGION_SOLVERS_KDR_TEMPLATE template < \
    typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM, \
    typename KERNEL_COORD_T, typename DOMAIN_COORD_T, typename RANGE_COORD_T>

#define LEGION_SOLVERS_KDR_TEMPLATE_ARGS \
    ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM, \
    KERNEL_COORD_T, DOMAIN_COORD_T, RANGE_COORD_T

// clang-format on

namespace LegionSolvers {


// clang-format off

template <typename FIELD_TYPE, int DIM, typename COORD_T>
using AffineReader = Legion::FieldAccessor<
    LEGION_READ_ONLY, FIELD_TYPE, DIM, COORD_T,
    Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
    LEGION_SOLVERS_CHECK_BOUNDS
>;

template <typename FIELD_TYPE, int DIM, typename COORD_T>
using AffineWriter = Legion::FieldAccessor<
    LEGION_WRITE_ONLY, FIELD_TYPE, DIM, COORD_T,
    Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
    LEGION_SOLVERS_CHECK_BOUNDS
>;

template <typename FIELD_TYPE, int DIM, typename COORD_T>
using AffineReaderWriter = Legion::FieldAccessor<
    LEGION_READ_WRITE, FIELD_TYPE, DIM, COORD_T,
    Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
    LEGION_SOLVERS_CHECK_BOUNDS
>;

template <typename FIELD_TYPE, int DIM, typename COORD_T>
using AffineSumAccessor = Legion::ReductionAccessor<
    Legion::SumReduction<FIELD_TYPE>, false, // non-exclusive
    DIM, COORD_T, Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
    LEGION_SOLVERS_CHECK_BOUNDS
>;

// clang-format on


enum class TaskFlags : std::uint8_t {
    LEAF = 0x01,
    INNER = 0x02,
    IDEMPOTENT = 0x04,
    REPLICABLE = 0x08,
}; // enum class TaskFlags

constexpr TaskFlags operator|(TaskFlags lhs, TaskFlags rhs) noexcept {
    return static_cast<TaskFlags>(
        static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs)
    );
}

constexpr bool operator&(TaskFlags lhs, TaskFlags rhs) noexcept {
    return static_cast<bool>(
        static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs)
    );
}


// clang-format off

template <void (*TASK_PTR)(const Legion::Task *,
                           const std::vector<Legion::PhysicalRegion> &,
                           Legion::Context, Legion::Runtime *)>
void preregister_task(Legion::TaskID task_id,
                      const std::string &task_name,
                      Legion::Processor::Kind proc_kind,
                      bool verbose = true) {
    if (verbose) {
        std::cout << "[LegionSolvers] Registering task " << task_name
                  << " with ID " << task_id << "." << std::endl;
    }
    Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    Legion::Runtime::preregister_task_variant<TASK_PTR>(
        registrar, task_name.c_str()
    );
}

template <void (*TASK_PTR)(const Legion::Task *,
                           const std::vector<Legion::PhysicalRegion> &,
                           Legion::Context, Legion::Runtime *)>
void preregister_task(Legion::TaskID task_id,
                      const std::string &task_name,
                      TaskFlags task_flags,
                      Legion::Processor::Kind proc_kind,
                      bool verbose = true) {
    if (verbose) {
        std::cout << "[LegionSolvers] Registering task " << task_name
                  << " with ID " << task_id << "." << std::endl;
    }
    Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    registrar.set_leaf(task_flags & TaskFlags::LEAF);
    registrar.set_inner(task_flags & TaskFlags::INNER);
    registrar.set_idempotent(task_flags & TaskFlags::IDEMPOTENT);
    registrar.set_replicable(task_flags & TaskFlags::REPLICABLE);
    Legion::Runtime::preregister_task_variant<TASK_PTR>(
        registrar, task_name.c_str()
    );
}

template <typename RETURN_T,
          RETURN_T (*TASK_PTR)(const Legion::Task *,
                               const std::vector<Legion::PhysicalRegion> &,
                               Legion::Context, Legion::Runtime *)>
void preregister_task(Legion::TaskID task_id,
                      const std::string &task_name,
                      Legion::Processor::Kind proc_kind,
                      bool verbose = true) {
    if (verbose) {
        std::cout << "[LegionSolvers] Registering task " << task_name
                  << " with ID " << task_id << "." << std::endl;
    }
    Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    if constexpr (std::is_void_v<RETURN_T>) {
        Legion::Runtime::preregister_task_variant<TASK_PTR>(
            registrar, task_name.c_str()
        );
    } else {
        Legion::Runtime::preregister_task_variant<RETURN_T, TASK_PTR>(
            registrar, task_name.c_str()
        );
    }
}

template <typename RETURN_T,
          RETURN_T (*TASK_PTR)(const Legion::Task *,
                               const std::vector<Legion::PhysicalRegion> &,
                               Legion::Context, Legion::Runtime *)>
void preregister_task(Legion::TaskID task_id,
                      const std::string &task_name,
                      TaskFlags task_flags,
                      Legion::Processor::Kind proc_kind,
                      bool verbose = true) {
    if (verbose) {
        std::cout << "[LegionSolvers] Registering task " << task_name
                  << " with ID " << task_id << "." << std::endl;
    }
    Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    registrar.set_leaf(task_flags & TaskFlags::LEAF);
    registrar.set_inner(task_flags & TaskFlags::INNER);
    registrar.set_idempotent(task_flags & TaskFlags::IDEMPOTENT);
    registrar.set_replicable(task_flags & TaskFlags::REPLICABLE);
    if constexpr (std::is_void_v<RETURN_T>) {
        Legion::Runtime::preregister_task_variant<TASK_PTR>(
            registrar, task_name.c_str()
        );
    } else {
        Legion::Runtime::preregister_task_variant<RETURN_T, TASK_PTR>(
            registrar, task_name.c_str()
        );
    }
}

// clang-format on


Legion::FieldSpace create_field_space(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::vector<std::size_t> &field_sizes,
    const std::vector<Legion::FieldID> &field_ids
);


void print_index_partition(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::string &name,
    Legion::IndexPartition index_partition
);


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LEGION_UTILITIES_HPP_INCLUDED

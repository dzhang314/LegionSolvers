#ifndef LEGION_SOLVERS_LEGION_UTILITIES_HPP
#define LEGION_SOLVERS_LEGION_UTILITIES_HPP

#include <cstddef> // for std::size_t
#include <utility> // for std::pair
#include <vector>  // for std::vector

#include <legion.h>


namespace LegionSolvers {


    template <int DIM>
    Legion::LogicalRegionT<DIM> create_region(
        Legion::IndexSpaceT<DIM> index_space,
        const std::vector<std::pair<std::size_t, Legion::FieldID>> &fields,
        Legion::Context ctx, Legion::Runtime *rt
    );


    template <void (*TASK_PTR)(const Legion::Task *,
                               const std::vector<Legion::PhysicalRegion> &,
                               Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id,
                              const std::string &task_name,
                              bool is_leaf, bool verbose) {
        if (verbose) {
            std::cout << "[LegionSolvers] Registering task " << task_name
                      << " with ID " << task_id << "." << std::endl;
        }
        Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
        registrar.add_constraint(
            Legion::ProcessorConstraint{Legion::Processor::LOC_PROC}
        );
        registrar.set_leaf(is_leaf);
        Legion::Runtime::preregister_task_variant<TASK_PTR>(
            registrar, task_name.c_str()
        );
    }


    template <typename RETURN_T,
              RETURN_T (*TASK_PTR)(const Legion::Task *,
                                   const std::vector<Legion::PhysicalRegion> &,
                                   Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id,
                              const std::string &task_name,
                              bool is_leaf, bool verbose) {
        if (verbose) {
            std::cout << "[LegionSolvers] Registering task " << task_name
                      << " with ID " << task_id << "." << std::endl;
        }
        Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
        registrar.add_constraint(
            Legion::ProcessorConstraint{Legion::Processor::LOC_PROC}
        );
        registrar.set_leaf(is_leaf);
        Legion::Runtime::preregister_task_variant<RETURN_T, TASK_PTR>(
            registrar, task_name.c_str()
        );
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LEGION_UTILITIES_HPP

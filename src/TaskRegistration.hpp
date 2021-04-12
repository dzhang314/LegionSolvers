#ifndef LEGION_SOLVERS_TASK_REGISTRATION_HPP
#define LEGION_SOLVERS_TASK_REGISTRATION_HPP

#include <iostream>
#include <string>

#include <legion.h>

#include "COOMatrixTasks.hpp"
#include "ExampleSystems.hpp"
#include "LegionSolversMapper.hpp"
#include "LinearAlgebraTasks.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <void (*TASK_PTR)(const Legion::Task *,
                               const std::vector<Legion::PhysicalRegion> &,
                               Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id,
                              const std::string &task_name,
                              bool is_leaf, bool verbose) {
        if (verbose) {
            std::cout << "Registering task " << task_name
                      << " with ID " << task_id << "." << std::endl;
        }
        Legion::TaskVariantRegistrar registrar(task_id, task_name.c_str());
        registrar.add_constraint(Legion::ProcessorConstraint{
            Legion::Processor::LOC_PROC});
        registrar.set_leaf(is_leaf);
        Legion::Runtime::preregister_task_variant<TASK_PTR>(
            registrar, task_name.c_str());
    }


    template <typename RETURN_T,
              RETURN_T (*TASK_PTR)(const Legion::Task *,
                                   const std::vector<Legion::PhysicalRegion> &,
                                   Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id,
                              const std::string &task_name,
                              bool is_leaf, bool verbose) {
        if (verbose) {
            std::cout << "Registering task " << task_name
                      << " with ID " << task_id << "." << std::endl;
        }
        Legion::TaskVariantRegistrar registrar(task_id, task_name.c_str());
        registrar.add_constraint(Legion::ProcessorConstraint{
            Legion::Processor::LOC_PROC});
        registrar.set_leaf(is_leaf);
        Legion::Runtime::preregister_task_variant<RETURN_T, TASK_PTR>(
            registrar, task_name.c_str());
    }


    template <template <typename, int...> typename TASK_CLASS, typename... TS>
    struct CartesianProductRegistrar;

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS>
    struct CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            preregister_cpu_task<TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                             TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                 std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                 ToString<IntList<MS...>>::value(),
                                                             is_leaf, verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, typename... TS, int... MS>
    struct CartesianProductRegistrar<TASK_CLASS, TypeList<T, TS...>, IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            preregister_cpu_task<TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                             TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                 std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                 ToString<IntList<MS...>>::value(),
                                                             is_leaf, verbose);
            CartesianProductRegistrar<TASK_CLASS, TypeList<TS...>, IntList<MS...>, IntList<>>::execute(is_leaf, verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, int... MS>
    struct CartesianProductRegistrar<TASK_CLASS, TypeList<>, IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int... NS>
    struct CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<0, NS...>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int N, int... NS>
    struct CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<N, NS...>> {
        static void execute(bool is_leaf, bool verbose) {
            CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<N - 1, NS...>>::execute(is_leaf, verbose);
            CartesianProductRegistrar<TASK_CLASS, T, IntList<MS..., N>, IntList<NS...>>::execute(is_leaf, verbose);
        }
    };


    template <template <typename, int...> typename TASK_CLASS, typename... TS>
    struct CartesianProductRegistrarRT;

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS>
    struct CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            preregister_cpu_task<T, TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                                TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                    ToString<IntList<MS...>>::value(),
                                                                is_leaf, verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, typename... TS, int... MS>
    struct CartesianProductRegistrarRT<TASK_CLASS, TypeList<T, TS...>, IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            preregister_cpu_task<T, TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                                TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                    ToString<IntList<MS...>>::value(),
                                                                is_leaf, verbose);
            CartesianProductRegistrarRT<TASK_CLASS, TypeList<TS...>, IntList<MS...>, IntList<>>::execute(is_leaf, verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, int... MS>
    struct CartesianProductRegistrarRT<TASK_CLASS, TypeList<>, IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int... NS>
    struct CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<0, NS...>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int N, int... NS>
    struct CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<N, NS...>> {
        static void execute(bool is_leaf, bool verbose) {
            CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<N - 1, NS...>>::execute(is_leaf, verbose);
            CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS..., N>, IntList<NS...>>::execute(is_leaf, verbose);
        }
    };


    struct ProjectionOneLevel final : public Legion::ProjectionFunctor {

        Legion::coord_t index;

        explicit ProjectionOneLevel(Legion::coord_t index) noexcept :
            index(index) {}

        virtual bool is_functional(void) const noexcept { return true; }

        virtual unsigned get_depth(void) const noexcept { return 0; }

        // using Legion::ProjectionFunctor::project;

        virtual Legion::LogicalRegion project(
                Legion::LogicalPartition upper_bound,
                const Legion::DomainPoint &point,
                const Legion::Domain &launch_domain) override {
            return runtime->get_logical_subregion_by_color(
                upper_bound, point[index]);
        }

    }; // struct ProjectionOneLevel


    Legion::Color GLOBAL_TILE_PARTITION_COLOR = 500;


    struct ProjectionTwoLevel final : public Legion::ProjectionFunctor {

        Legion::coord_t i;
        Legion::coord_t j;

        explicit ProjectionTwoLevel(Legion::coord_t i, Legion::coord_t j)
            noexcept : i(i), j(j) {}

        virtual bool is_functional(void) const noexcept { return true; }

        virtual unsigned get_depth(void) const noexcept { return 1; }

        virtual Legion::LogicalRegion project(
                Legion::LogicalPartition upper_bound,
                const Legion::DomainPoint &point,
                const Legion::Domain &launch_domain) override {
            const auto column = runtime->get_logical_subregion_by_color(
                upper_bound, point[i]);
            const auto column_partition = runtime->get_logical_partition_by_color(
                column, GLOBAL_TILE_PARTITION_COLOR);
            return runtime->get_logical_subregion_by_color(
                column_partition, point[j]);
        }

    }; // struct ProjectionTwoLevel


    void preregister_solver_tasks(bool verbose = false) {
        CartesianProductRegistrarRT<AdditionTask,
            LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<>>::execute(true, verbose);
        CartesianProductRegistrarRT<SubtractionTask,
            LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<>>::execute(true, verbose);
        CartesianProductRegistrarRT<NegationTask,
            LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<>>::execute(true, verbose);
        CartesianProductRegistrarRT<MultiplicationTask,
            LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<>>::execute(true, verbose);
        CartesianProductRegistrarRT<DivisionTask,
            LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<>>::execute(true, verbose);
        CartesianProductRegistrar<DummyTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<ConstantFillTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<RandomFillTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<CopyTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<AxpyTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<XpayTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrarRT<DotProductTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                    IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<
            COOMatvecTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
            IntList<LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<PrintVectorTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<
            COORmatvecTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
            IntList<LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<
            COOPrintTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
            IntList<LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM>>::execute(true, verbose);
        CartesianProductRegistrar<FillCOONegativeLaplacian1DTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<>>::execute(true, verbose);
        CartesianProductRegistrar<FillCOONegativeLaplacian2DTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<>>::execute(true, verbose);
        Legion::Runtime::add_registration_callback(mapper_registration_callback);
        Legion::Runtime::preregister_projection_functor(
            PFID_IJ_TO_I, new ProjectionOneLevel{0});
        Legion::Runtime::preregister_projection_functor(
            PFID_IJ_TO_J, new ProjectionOneLevel{1});
        Legion::Runtime::preregister_projection_functor(
            PFID_IJ_TO_IJ, new ProjectionTwoLevel{0, 1});
        Legion::Runtime::preregister_projection_functor(
            PFID_IJ_TO_JI, new ProjectionTwoLevel{1, 0});
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_REGISTRATION_HPP

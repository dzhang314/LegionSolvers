#ifndef LEGION_SOLVERS_TASK_REGISTRATION_HPP
#define LEGION_SOLVERS_TASK_REGISTRATION_HPP

#include <iostream>
#include <string>

#include <legion.h>

#include "COOMatrixTasks.hpp"
#include "LinearAlgebraTasks.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <void (*TASK_PTR)(
        const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id, const std::string &task_name, bool verbose = true) {
        if (verbose) { std::cout << "Registering task " << task_name << " with ID " << task_id << "." << std::endl; }
        Legion::TaskVariantRegistrar registrar(task_id, task_name.c_str());
        registrar.add_constraint(Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
        Legion::Runtime::preregister_task_variant<TASK_PTR>(registrar, task_name.c_str());
    }


    template <
        typename RETURN_T,
        RETURN_T (*TASK_PTR)(
            const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id, const std::string &task_name, bool verbose = true) {
        if (verbose) { std::cout << "Registering task " << task_name << " with ID " << task_id << "." << std::endl; }
        Legion::TaskVariantRegistrar registrar(task_id, task_name.c_str());
        registrar.add_constraint(Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
        Legion::Runtime::preregister_task_variant<RETURN_T, TASK_PTR>(registrar, task_name.c_str());
    }


    template <typename RETURN_T, template <int> typename TASK_CLASS>
    void preregister(const std::string &task_name) {
        preregister_cpu_task<RETURN_T, TASK_CLASS<1>::task>(TASK_CLASS<1>::task_id, (task_name + "_1").c_str());
        preregister_cpu_task<RETURN_T, TASK_CLASS<2>::task>(TASK_CLASS<2>::task_id, (task_name + "_2").c_str());
        preregister_cpu_task<RETURN_T, TASK_CLASS<3>::task>(TASK_CLASS<3>::task_id, (task_name + "_3").c_str());
    }


    template <template <typename, int...> typename TASK_CLASS, typename... TS>
    struct CartesianProductRegistrar;

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS>
    struct CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<>> {
        static void execute(bool verbose = true) {
            preregister_cpu_task<TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                             TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                 std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                 ToString<IntList<MS...>>::value(),
                                                             verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, typename... TS, int... MS>
    struct CartesianProductRegistrar<TASK_CLASS, TypeList<T, TS...>, IntList<MS...>, IntList<>> {
        static void execute(bool verbose = true) {
            preregister_cpu_task<TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                             TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                 std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                 ToString<IntList<MS...>>::value(),
                                                             verbose);
            CartesianProductRegistrar<TASK_CLASS, TypeList<TS...>, IntList<MS...>, IntList<>>::execute(verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, int... MS>
    struct CartesianProductRegistrar<TASK_CLASS, TypeList<>, IntList<MS...>, IntList<>> {
        static void execute(bool verbose = true) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int... NS>
    struct CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<0, NS...>> {
        static void execute(bool verbose = true) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int N, int... NS>
    struct CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<N, NS...>> {
        static void execute(bool verbose = true) {
            CartesianProductRegistrar<TASK_CLASS, T, IntList<MS...>, IntList<N - 1, NS...>>::execute(verbose);
            CartesianProductRegistrar<TASK_CLASS, T, IntList<MS..., N>, IntList<NS...>>::execute(verbose);
        }
    };


    template <template <typename, int...> typename TASK_CLASS, typename... TS>
    struct CartesianProductRegistrarRT;

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS>
    struct CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<>> {
        static void execute(bool verbose = true) {
            preregister_cpu_task<T, TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                                TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                    ToString<IntList<MS...>>::value(),
                                                                verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, typename... TS, int... MS>
    struct CartesianProductRegistrarRT<TASK_CLASS, TypeList<T, TS...>, IntList<MS...>, IntList<>> {
        static void execute(bool verbose = true) {
            preregister_cpu_task<T, TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                                TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                    ToString<IntList<MS...>>::value(),
                                                                verbose);
            CartesianProductRegistrarRT<TASK_CLASS, TypeList<TS...>, IntList<MS...>, IntList<>>::execute(verbose);
        }
    };

    template <template <typename, int...> typename TASK_CLASS, int... MS>
    struct CartesianProductRegistrarRT<TASK_CLASS, TypeList<>, IntList<MS...>, IntList<>> {
        static void execute(bool verbose = true) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int... NS>
    struct CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<0, NS...>> {
        static void execute(bool verbose = true) {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int N, int... NS>
    struct CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<N, NS...>> {
        static void execute(bool verbose = true) {
            CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS...>, IntList<N - 1, NS...>>::execute(verbose);
            CartesianProductRegistrarRT<TASK_CLASS, T, IntList<MS..., N>, IntList<NS...>>::execute(verbose);
        }
    };


    void preregister_solver_tasks(bool verbose = true) {
        CartesianProductRegistrarRT<AdditionTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>>::execute(
            verbose);
        CartesianProductRegistrarRT<SubtractionTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>>::execute(
            verbose);
        CartesianProductRegistrarRT<NegationTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>>::execute(
            verbose);
        CartesianProductRegistrarRT<MultiplicationTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>>::execute(
            verbose);
        CartesianProductRegistrarRT<DivisionTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>>::execute(
            verbose);
        preregister<bool, IsNonemptyTask>("is_nonempty");
        CartesianProductRegistrar<ConstantFillTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(verbose);
        CartesianProductRegistrar<CopyTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(verbose);
        CartesianProductRegistrar<AxpyTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(verbose);
        CartesianProductRegistrar<XpayTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                  IntList<LEGION_SOLVERS_MAX_DIM>>::execute(verbose);
        CartesianProductRegistrarRT<DotProductTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
                                    IntList<LEGION_SOLVERS_MAX_DIM>>::execute(verbose);
        CartesianProductRegistrar<
            CooMatvecTask, LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>,
            IntList<LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM, LEGION_SOLVERS_MAX_DIM>>::execute(verbose);
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_REGISTRATION_HPP

#ifndef LEGION_SOLVERS_TASK_REGISTRATION_HPP
#define LEGION_SOLVERS_TASK_REGISTRATION_HPP

#include <iostream>
#include <string>

#include <legion.h>

#include <Kokkos_Core.hpp>

#include "COOMatrixTasks.hpp"
#include "ExampleSystems.hpp"
#include "LegionSolversMapper.hpp"
#include "LinearAlgebraTasks.hpp"
#include "TaskIDs.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    /*
     * CartesianProductRegistrar<TaskClass, T, IntList<>, IntList<m, n, ...>>
     * is used to register a collection of Legion template tasks wrapped up
     * inside a class template TaskClass.
     *
     * CartesianProductRegistrar<TaskA, float,
                                 IntList<>, IntList<2, 3>>::execute()
     * registers the following six tasks:
     *
     *     TaskA<float, 1, 1>::task
     *     TaskA<float, 1, 2>::task
     *     TaskA<float, 1, 3>::task
     *     TaskA<float, 2, 1>::task
     *     TaskA<float, 2, 2>::task
     *     TaskA<float, 2, 3>::task
     *
     * CartesianProductRegistrar<TaskB, TypeList<float, double>,
                                 IntList<>, IntList<1, 1, 2>>::execute()
     * registers the following four tasks:
     *
     *     TaskB<float, 1, 1, 1>::task
     *     TaskB<float, 1, 1, 2>::task
     *     TaskB<double, 1, 1, 1>::task
     *     TaskB<double, 1, 1, 2>::task
     */
    template <template <typename, int...> typename TaskClass, typename... TS>
    struct CartesianProductRegistrar;

    // Base case: all dimensions instantiated, single type T.
    template <template <typename, int...> typename TaskClass,
              typename T, int... MS>
    struct CartesianProductRegistrar<TaskClass, T,
                                     IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            preregister_cpu_task<TaskClass<T, MS...>::task>(
                TaskClass<T, MS...>::task_id,
                TaskClass<T, MS...>::task_name() +
                    std::string{"_"} +
                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                    ToString<IntList<MS...>>::value(),
                is_leaf, verbose
            );
        }
    };

    // Base case: all dimensions instantiated, empty type list.
    template <template <typename, int...> typename TaskClass, int... MS>
    struct CartesianProductRegistrar<TaskClass, TypeList<>,
                                     IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    // Recursive case: all dimensions instantiated, non-empty type list.
    template <template <typename, int...> typename TaskClass,
              typename T, typename... TS, int... MS>
    struct CartesianProductRegistrar<TaskClass, TypeList<T, TS...>,
                                     IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            // Register first type...
            preregister_cpu_task<TaskClass<T, MS...>::task>(
                TaskClass<T, MS...>::task_id,
                TaskClass<T, MS...>::task_name() +
                    std::string{"_"} +
                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                    ToString<IntList<MS...>>::value(),
                is_leaf, verbose
            );
            // ...then recurse on remaining types.
            CartesianProductRegistrar<
                TaskClass, TypeList<TS...>, IntList<MS...>, IntList<>
            >::execute(is_leaf, verbose);
        }
    };

    // Base case: instantiating dimensions, first dimension exhausted.
    template <template <typename, int...> typename TaskClass,
              typename T, int... MS, int... NS>
    struct CartesianProductRegistrar<TaskClass, T,
                                     IntList<MS...>, IntList<0, NS...>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    // Recursive case: instantiating dimensions, recursing on first dimension.
    template <template <typename, int...> typename TaskClass,
              typename T, int... MS, int N, int... NS>
    struct CartesianProductRegistrar<TaskClass, T,
                                     IntList<MS...>, IntList<N, NS...>> {
        static void execute(bool is_leaf, bool verbose) {
            // Recurse on smaller values of first dimension...
            CartesianProductRegistrar<
                TaskClass, T, IntList<MS...>, IntList<N - 1, NS...>
            >::execute(is_leaf, verbose);
            // ...then instantiate current value of first dimension.
            CartesianProductRegistrar<
                TaskClass, T, IntList<MS..., N>, IntList<NS...>
            >::execute(is_leaf, verbose);
        }
    };


    template <template <typename> typename TaskClass, typename... TS>
    struct ScalarTaskRegistrar;

    // Base case: empty type list.
    template <template <typename> typename TaskClass>
    struct ScalarTaskRegistrar<TaskClass, TypeList<>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    // Recursive case: non-empty type list.
    template <template <typename> typename TaskClass,
              typename T, typename... TS>
    struct ScalarTaskRegistrar<TaskClass, TypeList<T, TS...>> {
        static void execute(bool is_leaf, bool verbose) {
            // Register first type...
            preregister_cpu_task<T, TaskClass<T>::task>(
                TaskClass<T>::task_id,
                TaskClass<T>::task_name() +
                    std::string{"_"} +
                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()},
                is_leaf, verbose
            );
            // ...then recurse on remaining types.
            ScalarTaskRegistrar<
                TaskClass, TypeList<TS...>
            >::execute(is_leaf, verbose);
        }
    };


    template <template <typename> typename TaskClass>
    void preregister_scalar_leaf_task(bool verbose) {
        ScalarTaskRegistrar<
            TaskClass, LEGION_SOLVERS_SUPPORTED_TYPES,
        >::execute(true, verbose);
    }


    template <template <typename, int> typename TaskClass>
    void preregister_vector_leaf_task(bool verbose) {
        CartesianProductRegistrar<
            TaskClass,
            LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<LEGION_SOLVERS_MAX_DIM>
        >::execute(true, verbose);
    }


    template <template <typename, int> typename TaskClass>
    void preregister_vector_kokkos_task(bool verbose) {
        TaskClass<float, 1>::preregister(verbose);
        TaskClass<float, 2>::preregister(verbose);
        TaskClass<float, 3>::preregister(verbose);
        TaskClass<double, 1>::preregister(verbose);
        TaskClass<double, 2>::preregister(verbose);
        TaskClass<double, 3>::preregister(verbose);
    }


    template <template <typename, int, int, int> typename TaskClass>
    void preregister_matrix_leaf_task(bool verbose) {
        CartesianProductRegistrar<
            TaskClass,
            LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<LEGION_SOLVERS_MAX_DIM,
                               LEGION_SOLVERS_MAX_DIM,
                               LEGION_SOLVERS_MAX_DIM>
        >::execute(true, verbose);
    }


    void preregister_solver_tasks(bool verbose = false) {

        preregister_scalar_leaf_task<AdditionTask      >(verbose);
        preregister_scalar_leaf_task<SubtractionTask   >(verbose);
        preregister_scalar_leaf_task<NegationTask      >(verbose);
        preregister_scalar_leaf_task<MultiplicationTask>(verbose);
        preregister_scalar_leaf_task<DivisionTask      >(verbose);

        preregister_vector_leaf_task<DummyTask       >(verbose);
        preregister_vector_leaf_task<RandomFillTask  >(verbose);
        preregister_vector_leaf_task<PrintVectorTask >(verbose);

        AxpyTask<float, 1>::preregister(verbose);
        AxpyTask<float, 2>::preregister(verbose);
        AxpyTask<float, 3>::preregister(verbose);
        AxpyTask<double, 1>::preregister(verbose);
        AxpyTask<double, 2>::preregister(verbose);
        AxpyTask<double, 3>::preregister(verbose);

        XpayTask<float, 1>::preregister(verbose);
        XpayTask<float, 2>::preregister(verbose);
        XpayTask<float, 3>::preregister(verbose);
        XpayTask<double, 1>::preregister(verbose);
        XpayTask<double, 2>::preregister(verbose);
        XpayTask<double, 3>::preregister(verbose);

        DotProductTask<float, 1>::preregister(verbose);
        DotProductTask<float, 2>::preregister(verbose);
        DotProductTask<float, 3>::preregister(verbose);
        DotProductTask<double, 1>::preregister(verbose);
        DotProductTask<double, 2>::preregister(verbose);
        DotProductTask<double, 3>::preregister(verbose);

        COOMatvecTask<float, 1, 1, 1>::preregister(verbose);
        COOMatvecTask<float, 1, 1, 2>::preregister(verbose);
        COOMatvecTask<float, 1, 1, 3>::preregister(verbose);
        COOMatvecTask<float, 1, 2, 1>::preregister(verbose);
        COOMatvecTask<float, 1, 2, 2>::preregister(verbose);
        COOMatvecTask<float, 1, 2, 3>::preregister(verbose);
        COOMatvecTask<float, 1, 3, 1>::preregister(verbose);
        COOMatvecTask<float, 1, 3, 2>::preregister(verbose);
        COOMatvecTask<float, 1, 3, 3>::preregister(verbose);
        COOMatvecTask<float, 2, 1, 1>::preregister(verbose);
        COOMatvecTask<float, 2, 1, 2>::preregister(verbose);
        COOMatvecTask<float, 2, 1, 3>::preregister(verbose);
        COOMatvecTask<float, 2, 2, 1>::preregister(verbose);
        COOMatvecTask<float, 2, 2, 2>::preregister(verbose);
        COOMatvecTask<float, 2, 2, 3>::preregister(verbose);
        COOMatvecTask<float, 2, 3, 1>::preregister(verbose);
        COOMatvecTask<float, 2, 3, 2>::preregister(verbose);
        COOMatvecTask<float, 2, 3, 3>::preregister(verbose);
        COOMatvecTask<float, 3, 1, 1>::preregister(verbose);
        COOMatvecTask<float, 3, 1, 2>::preregister(verbose);
        COOMatvecTask<float, 3, 1, 3>::preregister(verbose);
        COOMatvecTask<float, 3, 2, 1>::preregister(verbose);
        COOMatvecTask<float, 3, 2, 2>::preregister(verbose);
        COOMatvecTask<float, 3, 2, 3>::preregister(verbose);
        COOMatvecTask<float, 3, 3, 1>::preregister(verbose);
        COOMatvecTask<float, 3, 3, 2>::preregister(verbose);
        COOMatvecTask<float, 3, 3, 3>::preregister(verbose);
        COOMatvecTask<double, 1, 1, 1>::preregister(verbose);
        COOMatvecTask<double, 1, 1, 2>::preregister(verbose);
        COOMatvecTask<double, 1, 1, 3>::preregister(verbose);
        COOMatvecTask<double, 1, 2, 1>::preregister(verbose);
        COOMatvecTask<double, 1, 2, 2>::preregister(verbose);
        COOMatvecTask<double, 1, 2, 3>::preregister(verbose);
        COOMatvecTask<double, 1, 3, 1>::preregister(verbose);
        COOMatvecTask<double, 1, 3, 2>::preregister(verbose);
        COOMatvecTask<double, 1, 3, 3>::preregister(verbose);
        COOMatvecTask<double, 2, 1, 1>::preregister(verbose);
        COOMatvecTask<double, 2, 1, 2>::preregister(verbose);
        COOMatvecTask<double, 2, 1, 3>::preregister(verbose);
        COOMatvecTask<double, 2, 2, 1>::preregister(verbose);
        COOMatvecTask<double, 2, 2, 2>::preregister(verbose);
        COOMatvecTask<double, 2, 2, 3>::preregister(verbose);
        COOMatvecTask<double, 2, 3, 1>::preregister(verbose);
        COOMatvecTask<double, 2, 3, 2>::preregister(verbose);
        COOMatvecTask<double, 2, 3, 3>::preregister(verbose);
        COOMatvecTask<double, 3, 1, 1>::preregister(verbose);
        COOMatvecTask<double, 3, 1, 2>::preregister(verbose);
        COOMatvecTask<double, 3, 1, 3>::preregister(verbose);
        COOMatvecTask<double, 3, 2, 1>::preregister(verbose);
        COOMatvecTask<double, 3, 2, 2>::preregister(verbose);
        COOMatvecTask<double, 3, 2, 3>::preregister(verbose);
        COOMatvecTask<double, 3, 3, 1>::preregister(verbose);
        COOMatvecTask<double, 3, 3, 2>::preregister(verbose);
        COOMatvecTask<double, 3, 3, 3>::preregister(verbose);

        // preregister_matrix_leaf_task<COORmatvecTask>(verbose);
        // preregister_matrix_leaf_task<COOPrintTask  >(verbose);

        CartesianProductRegistrar<
            FillCOONegativeLaplacian1DTask,
            LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>
        >::execute(true, verbose);

        CartesianProductRegistrar<
            FillCOONegativeLaplacian2DTask,
            LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>
        >::execute(true, verbose);

        Legion::Runtime::add_registration_callback(
            mapper_registration_callback
        );

        Legion::Runtime::preregister_projection_functor(
            PFID_KDR_TO_K, new ProjectionOneLevel{0}
        );
        Legion::Runtime::preregister_projection_functor(
            PFID_KDR_TO_D, new ProjectionOneLevel{1}
        );
        Legion::Runtime::preregister_projection_functor(
            PFID_KDR_TO_R, new ProjectionOneLevel{2}
        );
        Legion::Runtime::preregister_projection_functor(
            PFID_KDR_TO_DR, new ProjectionTwoLevel{1, 2}
        );
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_REGISTRATION_HPP

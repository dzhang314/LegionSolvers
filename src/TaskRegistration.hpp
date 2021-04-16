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
#include "UtilityTasks.hpp"


namespace LegionSolvers {


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


    template <template <typename> typename PORTABLE_KOKKOS_TASK>
    void preregister_kokkos_task(Legion::TaskID task_id,
                                 const std::string &task_name,
                                 bool is_leaf, bool verbose) {

        #ifdef KOKKOS_ENABLE_SERIAL
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos CPU task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                Legion::Processor::LOC_PROC
            });
            registrar.set_leaf(is_leaf);
            Legion::Runtime::preregister_task_variant<
                PORTABLE_KOKKOS_TASK<Kokkos::Serial>::task_body
            >(registrar, task_name.c_str());
        }
        #endif

        #ifdef KOKKOS_ENABLE_OPENMP
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos OpenMP task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                #ifdef REALM_USE_OPENMP
                    Legion::Processor::OMP_PROC
                #else
                    Legion::Processor::LOC_PROC
                #endif
            });
            registrar.set_leaf(is_leaf);
            Legion::Runtime::preregister_task_variant<
                PORTABLE_KOKKOS_TASK<Kokkos::OpenMP>::task_body
            >(registrar, task_name.c_str());
        }
        #endif

        #if defined(KOKKOS_ENABLE_CUDA) and defined(REALM_USE_CUDA)
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos GPU task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                Legion::Processor::TOC_PROC
            });
            registrar.set_leaf(is_leaf);
            Legion::Runtime::preregister_task_variant<
                PORTABLE_KOKKOS_TASK<Kokkos::Cuda>::task_body
            >(registrar, task_name.c_str());
        }
        #endif
    }


    template <typename ReturnType,
              template <typename, typename, int...> typename TaskClass,
              typename T, int... NS>
    void preregister_kokkos_task(Legion::TaskID task_id,
                                 const std::string &task_name,
                                 bool is_leaf, bool verbose) {

        #ifdef KOKKOS_ENABLE_SERIAL
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos CPU task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                Legion::Processor::LOC_PROC
            });
            registrar.set_leaf(is_leaf);
            Legion::Runtime::preregister_task_variant<
                ReturnType, TaskClass<Kokkos::Serial, T, NS...>::task_body
            >(registrar, task_name.c_str());
        }
        #endif

        #ifdef KOKKOS_ENABLE_OPENMP
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos OpenMP task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                #ifdef REALM_USE_OPENMP
                    Legion::Processor::OMP_PROC
                #else
                    Legion::Processor::LOC_PROC
                #endif
            });
            registrar.set_leaf(is_leaf);
            Legion::Runtime::preregister_task_variant<
                ReturnType, TaskClass<Kokkos::OpenMP, T, NS...>::task_body
            >(registrar, task_name.c_str());
        }
        #endif

        #if defined(KOKKOS_ENABLE_CUDA) and defined(REALM_USE_CUDA)
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos GPU task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                Legion::Processor::TOC_PROC
            });
            registrar.set_leaf(is_leaf);
            Legion::Runtime::preregister_task_variant<
                ReturnType, TaskClass<Kokkos::Cuda, T, NS...>::task_body
            >(registrar, task_name.c_str());
        }
        #endif
    }


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


    template <template <typename, int...> typename TaskClass, typename... TS>
    struct ScalarTaskRegistrar;

    // Base case: all dimensions instantiated, empty type list.
    template <template <typename, int...> typename TaskClass>
    struct ScalarTaskRegistrar<TaskClass, TypeList<>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    // Recursive case: all dimensions instantiated, non-empty type list.
    template <template <typename, int...> typename TaskClass,
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


    /*
     * KokkosTaskRegistrarRT is just like CartesianProductRegistrar
     * except it registers Kokkos tasks that return T, rather than void.
     */
    template <template <typename, typename, int...> typename TaskClass,
              typename... TS>
    struct KokkosTaskRegistrarRT;

    // Base case: all dimensions instantiated, single type T.
    template <template <typename, typename, int...> typename TaskClass,
              typename T, int... MS>
    struct KokkosTaskRegistrarRT<TaskClass, T,
                                 IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            preregister_kokkos_task<T, TaskClass, T, MS...>(
                TaskClass<Kokkos::DefaultExecutionSpace, T, MS...>::task_id,
                TaskClass<Kokkos::DefaultExecutionSpace, T, MS...>::task_name() +
                    std::string{"_"} +
                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                    ToString<IntList<MS...>>::value(),
                is_leaf, verbose
            );
        }
    };

    // Base case: all dimensions instantiated, empty type list.
    template <template <typename, typename, int...> typename TaskClass,
              int... MS>
    struct KokkosTaskRegistrarRT<TaskClass, TypeList<>,
                                 IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    // Recursive case: all dimensions instantiated, non-empty type list.
    template <template <typename, typename, int...> typename TaskClass,
              typename T, typename... TS, int... MS>
    struct KokkosTaskRegistrarRT<TaskClass, TypeList<T, TS...>,
                                 IntList<MS...>, IntList<>> {
        static void execute(bool is_leaf, bool verbose) {
            // Register first type...
            preregister_kokkos_task<T, TaskClass, T, MS...>(
                TaskClass<Kokkos::DefaultExecutionSpace, T, MS...>::task_id,
                TaskClass<Kokkos::DefaultExecutionSpace, T, MS...>::task_name() +
                    std::string{"_"} +
                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                    ToString<IntList<MS...>>::value(),
                is_leaf, verbose
            );
            // ...then recurse on remaining types.
            KokkosTaskRegistrarRT<
                TaskClass, TypeList<TS...>, IntList<MS...>, IntList<>
            >::execute(is_leaf, verbose);
        }
    };

    // Base case: instantiating dimensions, first dimension exhausted.
    template <template <typename, typename, int...> typename TaskClass,
              typename T, int... MS, int... NS>
    struct KokkosTaskRegistrarRT<TaskClass, T,
                                 IntList<MS...>, IntList<0, NS...>> {
        static void execute(bool is_leaf, bool verbose) {}
    };

    // Recursive case: instantiating dimensions, recursing on first dimension.
    template <template <typename, typename, int...> typename TaskClass,
              typename T, int... MS, int N, int... NS>
    struct KokkosTaskRegistrarRT<TaskClass, T,
                                 IntList<MS...>, IntList<N, NS...>> {
        static void execute(bool is_leaf, bool verbose) {
            // Recurse on smaller values of first dimension...
            KokkosTaskRegistrarRT<
                TaskClass, T, IntList<MS...>, IntList<N - 1, NS...>
            >::execute(is_leaf, verbose);
            // ...then instantiate current value of first dimension.
            KokkosTaskRegistrarRT<
                TaskClass, T, IntList<MS..., N>, IntList<NS...>
            >::execute(is_leaf, verbose);
        }
    };


    struct ProjectionOneLevel final : public Legion::ProjectionFunctor {

        Legion::coord_t index;

        explicit ProjectionOneLevel(Legion::coord_t index) noexcept :
            index(index) {}

        virtual bool is_functional(void) const noexcept { return true; }

        virtual unsigned get_depth(void) const noexcept { return 0; }

        using Legion::ProjectionFunctor::project;

        virtual Legion::LogicalRegion project(
                Legion::LogicalPartition upper_bound,
                const Legion::DomainPoint &point,
                const Legion::Domain &launch_domain) override {
            return runtime->get_logical_subregion_by_color(
                upper_bound, point[index]
            );
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

        using Legion::ProjectionFunctor::project;

        virtual Legion::LogicalRegion project(
                Legion::LogicalPartition upper_bound,
                const Legion::DomainPoint &point,
                const Legion::Domain &launch_domain) override {
            const auto column = runtime->get_logical_subregion_by_color(
                upper_bound, point[i]
            );
            const auto partition = runtime->get_logical_partition_by_color(
                column, GLOBAL_TILE_PARTITION_COLOR
            );
            return runtime->get_logical_subregion_by_color(partition, point[j]);
        }

    }; // struct ProjectionTwoLevel


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
        preregister_vector_leaf_task<ConstantFillTask>(verbose);
        preregister_vector_leaf_task<RandomFillTask  >(verbose);
        preregister_vector_leaf_task<CopyTask        >(verbose);
        preregister_vector_leaf_task<AxpyTask        >(verbose);
        preregister_vector_leaf_task<XpayTask        >(verbose);
        preregister_vector_leaf_task<PrintVectorTask >(verbose);

        KokkosTaskRegistrarRT<
            DotProductTask, LEGION_SOLVERS_SUPPORTED_TYPES,
            IntList<>, IntList<LEGION_SOLVERS_MAX_DIM>
        >::execute(true, verbose);

        preregister_matrix_leaf_task<COOMatvecTask >(verbose);
        preregister_matrix_leaf_task<COORmatvecTask>(verbose);
        preregister_matrix_leaf_task<COOPrintTask  >(verbose);

        CartesianProductRegistrar<
            FillCOONegativeLaplacian1DTask,
            LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>
        >::execute(true, verbose);

        CartesianProductRegistrar<
            FillCOONegativeLaplacian2DTask,
            LEGION_SOLVERS_SUPPORTED_TYPES, IntList<>, IntList<>
        >::execute(true, verbose);

        Legion::Runtime::add_registration_callback(
            mapper_registration_callback);

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

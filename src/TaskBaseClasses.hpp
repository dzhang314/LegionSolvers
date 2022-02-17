#ifndef LEGION_SOLVERS_TASK_BASE_CLASSES_HPP
#define LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

#include <iostream>
#include <string>
#include <typeinfo>

#include <legion.h>

#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "MetaprogrammingUtilities.hpp"


namespace LegionSolvers {


    template <typename T> constexpr const char *LEGION_SOLVERS_TYPE_NAME   () { return typeid(T).name(); }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<__half     >() { return "half"          ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<float      >() { return "float"         ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<double     >() { return "double"        ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<long double>() { return "longdouble"    ; }
    // template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<__float128 >() { return "float128"      ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<int        >() { return "int"           ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<unsigned   >() { return "unsigned"      ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<long long  >() { return "longlong"      ; }


    template <typename T>
    constexpr int LEGION_SOLVERS_INDEX_TYPE_INDEX = ListIndex<LEGION_SOLVERS_SUPPORTED_INDEX_TYPES, T>::value;
    template <typename T>
    constexpr int LEGION_SOLVERS_ENTRY_TYPE_INDEX = ListIndex<LEGION_SOLVERS_SUPPORTED_ENTRY_TYPES, T>::value;
    constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES = ListLength<LEGION_SOLVERS_SUPPORTED_INDEX_TYPES>::value;
    constexpr int LEGION_SOLVERS_NUM_ENTRY_TYPES = ListLength<LEGION_SOLVERS_SUPPORTED_ENTRY_TYPES>::value;
    constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_0 = 1;
    constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_1 = LEGION_SOLVERS_NUM_INDEX_TYPES * LEGION_SOLVERS_NUM_INDEX_TYPES_0;
    constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_2 = LEGION_SOLVERS_NUM_INDEX_TYPES * LEGION_SOLVERS_NUM_INDEX_TYPES_1;
    constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_3 = LEGION_SOLVERS_NUM_INDEX_TYPES * LEGION_SOLVERS_NUM_INDEX_TYPES_2;
    constexpr int LEGION_SOLVERS_MAX_DIM_0 = 1;
    constexpr int LEGION_SOLVERS_MAX_DIM_1 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_0;
    constexpr int LEGION_SOLVERS_MAX_DIM_2 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_1;
    constexpr int LEGION_SOLVERS_MAX_DIM_3 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_2;
    constexpr int LEGION_SOLVERS_TASK_BLOCK_SIZE = LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM_3 * LEGION_SOLVERS_NUM_INDEX_TYPES_3;


    template <Legion::TaskID BLOCK_ID,
              template <typename> typename TaskClass,
              typename T>
    struct TaskT {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

        static std::string task_name() {
            return std::string{TaskClass<T>::task_base_name} +
                   '_' + LEGION_SOLVERS_TYPE_NAME<T>();
        }

        static void preregister_cpu(bool verbose) {
            preregister_cpu_task<
                typename TaskClass<T>::return_type,
                TaskClass<T>::task_body
            >(
                task_id, task_name(), TaskClass<T>::is_replicable,
                TaskClass<T>::is_inner, TaskClass<T>::is_leaf,
                verbose
            );
        }

        static void announce_cpu(Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << " on processor " << proc << '.' << std::endl;
        }

        static void preregister_kokkos(bool verbose) {
            preregister_kokkos_task<
                typename TaskClass<T>::return_type,
                TaskClass<T>::template KokkosTaskTemplate
            >(
                task_id, task_name(), TaskClass<T>::is_replicable,
                TaskClass<T>::is_inner, TaskClass<T>::is_leaf,
                verbose
            );
        }

    }; // struct TaskT


    template <Legion::TaskID BLOCK_ID,
              template <int> typename TaskClass,
              int N>
    struct TaskD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            (N - 1);

        static std::string task_name() {
            return std::string{TaskClass<N>::task_base_name} +
                   '_' + std::to_string(N);
        }

        static void preregister_cpu(bool verbose) {
            preregister_cpu_task<
                typename TaskClass<N>::return_type,
                TaskClass<N>::task_body
            >(
                task_id, task_name(), TaskClass<N>::is_replicable,
                TaskClass<N>::is_inner, TaskClass<N>::is_leaf,
                verbose
            );
        }

        static void announce_cpu(Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << " on processor " << proc << '.' << std::endl;
        }

        static void preregister_kokkos(bool verbose) {
            preregister_kokkos_task<
                typename TaskClass<N>::return_type,
                TaskClass<N>::template KokkosTaskTemplate
            >(
                task_id, task_name(), TaskClass<N>::is_replicable,
                TaskClass<N>::is_inner, TaskClass<N>::is_leaf,
                verbose
            );
        }

    }; // struct TaskD


    template <Legion::TaskID BLOCK_ID,
              template <int> typename TaskClass>
    struct TaskD<BLOCK_ID, TaskClass, 0> {

        static constexpr Legion::TaskID task_id(int N) {
            return (LEGION_SOLVERS_TASK_ID_ORIGIN +
                    LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                    (N - 1));
        }

    }; // struct TaskD<BLOCK_ID, TaskClass, 0>


    template <Legion::TaskID BLOCK_ID,
              template <typename, int> typename TaskClass,
              typename T, int N>
    struct TaskTD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            LEGION_SOLVERS_NUM_ENTRY_TYPES * (N - 1) +
            LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

        static std::string task_name() {
            return std::string{TaskClass<T, N>::task_base_name} +
                   '_' + LEGION_SOLVERS_TYPE_NAME<T>() +
                   '_' + std::to_string(N);
        }

        static void preregister_cpu(bool verbose) {
            preregister_cpu_task<
                typename TaskClass<T, N>::return_type,
                TaskClass<T, N>::task_body
            >(
                task_id, task_name(), TaskClass<T, N>::is_replicable,
                TaskClass<T, N>::is_inner, TaskClass<T, N>::is_leaf,
                verbose
            );
        }

        static void announce_cpu(Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << " on processor " << proc << '.' << std::endl;
        }

        static void announce_cpu(Legion::DomainPoint index_point,
                                 Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << ", index point " << index_point
                      << ", on processor " << proc << '.' << std::endl;
        }

        static void preregister_kokkos(bool verbose) {
            preregister_kokkos_task<
                typename TaskClass<T, N>::return_type,
                TaskClass<T, N>::template KokkosTaskTemplate
            >(
                task_id, task_name(), TaskClass<T, N>::is_replicable,
                TaskClass<T, N>::is_inner, TaskClass<T, N>::is_leaf,
                verbose
            );
        }

        static void announce_kokkos(Legion::DomainPoint index_point,
                                    const std::type_info &execution_space,
                                    Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running Kokkos task " << task_name()
                      << ", index point " << index_point
                      << ", on processor " << proc
                      << " (kind " << proc.kind()
                      << ", " << execution_space.name() << ")." << std::endl;
        }

    }; // struct TaskTD


    template <Legion::TaskID BLOCK_ID,
              template <typename, int> typename TaskClass,
              typename T>
    struct TaskTD<BLOCK_ID, TaskClass, T, 0> {

        static constexpr Legion::TaskID task_id(int N) {
            return (LEGION_SOLVERS_TASK_ID_ORIGIN +
                    LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                    LEGION_SOLVERS_NUM_ENTRY_TYPES * (N - 1) +
                    LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>);
        }

    }; // struct TaskTD<BLOCK_ID, TaskClass, T, 0>


    template <Legion::TaskID BLOCK_ID,
              template <typename, int, typename> typename TaskClass,
              typename T, int N, typename I>
    struct TaskTDI {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_INDEX_TYPE_INDEX<I> +
            LEGION_SOLVERS_NUM_ENTRY_TYPES * (N - 1) +
            LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

        static std::string task_name() {
            return std::string{TaskClass<T, N, I>::task_base_name} +
                   '_' + LEGION_SOLVERS_TYPE_NAME<T>() +
                   '_' + std::to_string(N) +
                   '_' + LEGION_SOLVERS_TYPE_NAME<I>();
        }

        static void preregister_cpu(bool verbose) {
            preregister_cpu_task<
                typename TaskClass<T, N, I>::return_type,
                TaskClass<T, N, I>::task_body
            >(
                task_id, task_name(), TaskClass<T, N, I>::is_replicable,
                TaskClass<T, N, I>::is_inner, TaskClass<T, N, I>::is_leaf,
                verbose
            );
        }

        static void announce_cpu(Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << " on processor " << proc << '.' << std::endl;
        }

        static void announce_cpu(Legion::DomainPoint index_point,
                                 Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << ", index point " << index_point
                      << ", on processor " << proc << '.' << std::endl;
        }

        static void preregister_kokkos(bool verbose) {
            preregister_kokkos_task<
                typename TaskClass<T, N, I>::return_type,
                TaskClass<T, N, I>::template KokkosTaskTemplate
            >(
                task_id, task_name(), TaskClass<T, N, I>::is_replicable,
                TaskClass<T, N, I>::is_inner, TaskClass<T, N, I>::is_leaf,
                verbose
            );
        }

        static void announce_kokkos(Legion::DomainPoint index_point,
                                    const std::type_info &execution_space,
                                    Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running Kokkos task " << task_name()
                      << ", index point " << index_point
                      << ", on processor " << proc
                      << " (kind " << proc.kind()
                      << ", " << execution_space.name() << ")." << std::endl;
        }

    }; // struct TaskTDI


    template <Legion::TaskID BLOCK_ID,
              template <typename, int, typename> typename TaskClass,
              typename T>
    struct TaskTDI<BLOCK_ID, TaskClass, T, 0, void> {

        static constexpr Legion::TaskID task_id(int N, int I) {
            return (LEGION_SOLVERS_TASK_ID_ORIGIN +
                    LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                    LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM * I +
                    LEGION_SOLVERS_NUM_ENTRY_TYPES * (N - 1) +
                    LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>);
        }

        static Legion::TaskID task_id(Legion::IndexSpace index_space) {
            return task_id(
                index_space.get_dim(),
                index_space.get_type_tag() - 256 * index_space.get_dim()
            );
        }

    }; // struct TaskTDI<BLOCK_ID, TaskClass, T, 0, void>


    template <Legion::TaskID BLOCK_ID,
              template <typename, int, int, int> typename TaskClass,
              typename T, int N1, int N2, int N3>
    struct TaskTDDD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            LEGION_SOLVERS_MAX_DIM_2 * LEGION_SOLVERS_NUM_ENTRY_TYPES * (N1 - 1) +
            LEGION_SOLVERS_MAX_DIM_1 * LEGION_SOLVERS_NUM_ENTRY_TYPES * (N2 - 1) +
            LEGION_SOLVERS_MAX_DIM_0 * LEGION_SOLVERS_NUM_ENTRY_TYPES * (N3 - 1) +
            LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

        static std::string task_name() {
            return std::string{TaskClass<T, N1, N2, N3>::task_base_name} +
                   '_' + LEGION_SOLVERS_TYPE_NAME<T>() +
                   '_' + std::to_string(N1) +
                   '_' + std::to_string(N2) +
                   '_' + std::to_string(N3);
        }

        static void preregister_cpu(bool verbose) {
            preregister_cpu_task<
                typename TaskClass<T, N1, N2, N3>::return_type,
                TaskClass<T, N1, N2, N3>::task_body
            >(
                task_id, task_name(),
                TaskClass<T, N1, N2, N3>::is_replicable,
                TaskClass<T, N1, N2, N3>::is_inner,
                TaskClass<T, N1, N2, N3>::is_leaf,
                verbose
            );
        }

        static void announce_cpu(Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << " on processor " << proc << '.' << std::endl;
        }

        static void announce_cpu(Legion::DomainPoint index_point,
                                 Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running CPU task " << task_name()
                      << ", index point " << index_point
                      << ", on processor " << proc << '.' << std::endl;
        }

        static void preregister_kokkos(bool verbose) {
            preregister_kokkos_task<
                typename TaskClass<T, N1, N2, N3>::return_type,
                TaskClass<T, N1, N2, N3>::template KokkosTaskTemplate
            >(
                task_id, task_name(),
                TaskClass<T, N1, N2, N3>::is_replicable,
                TaskClass<T, N1, N2, N3>::is_inner,
                TaskClass<T, N1, N2, N3>::is_leaf,
                verbose
            );
        }

        static void announce_kokkos(Legion::DomainPoint index_point,
                                    const std::type_info &execution_space,
                                    Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running Kokkos task " << task_name()
                      << ", index point " << index_point
                      << ", on processor " << proc
                      << " (kind " << proc.kind()
                      << ", " << execution_space.name() << ")." << std::endl;
        }

    }; // struct TaskTDDD


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

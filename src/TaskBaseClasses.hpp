#ifndef LEGION_SOLVERS_TASK_BASE_CLASSES_HPP_INCLUDED
#define LEGION_SOLVERS_TASK_BASE_CLASSES_HPP_INCLUDED

#include <string> // for std::string

#include <legion.h> // for Legion::*

#include "LibraryOptions.hpp"           // for LEGION_SOLVERS_USE_*
#include "MetaprogrammingUtilities.hpp" // for TypeList, ListIndex, ...
#include "TaskIDs.hpp"                  // for NUM_META_TASK_IDS

namespace LegionSolvers {


// clang-format off
using LEGION_SOLVERS_SUPPORTED_INDEX_TYPES = TypeList<

    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        int,
    #else
        void,
    #endif // LEGION_SOLVERS_USE_S32_INDICES

    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        unsigned,
    #else
        void,
    #endif // LEGION_SOLVERS_USE_U32_INDICES

    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        long long,
    #else
        void,
    #endif // LEGION_SOLVERS_USE_S64_INDICES

    void
>;
// clang-format on


// clang-format off
using LEGION_SOLVERS_SUPPORTED_ENTRY_TYPES = TypeList<

    #ifdef LEGION_SOLVERS_USE_F32
        float,
    #endif // LEGION_SOLVERS_USE_F32

    #ifdef LEGION_SOLVERS_USE_F64
        double,
    #endif // LEGION_SOLVERS_USE_F64

    void
>;
// clang-format on


template <typename T>
constexpr int LEGION_SOLVERS_INDEX_TYPE_INDEX =
    ListIndex<LEGION_SOLVERS_SUPPORTED_INDEX_TYPES, T>::value;

template <typename T>
constexpr int LEGION_SOLVERS_ENTRY_TYPE_INDEX =
    ListIndex<LEGION_SOLVERS_SUPPORTED_ENTRY_TYPES, T>::value;

constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES =
    ListLength<LEGION_SOLVERS_SUPPORTED_INDEX_TYPES>::value;

constexpr int LEGION_SOLVERS_NUM_ENTRY_TYPES =
    ListLength<LEGION_SOLVERS_SUPPORTED_ENTRY_TYPES>::value;

constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_0 = 1;
constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_1 =
    LEGION_SOLVERS_NUM_INDEX_TYPES * LEGION_SOLVERS_NUM_INDEX_TYPES_0;
constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_2 =
    LEGION_SOLVERS_NUM_INDEX_TYPES * LEGION_SOLVERS_NUM_INDEX_TYPES_1;
constexpr int LEGION_SOLVERS_NUM_INDEX_TYPES_3 =
    LEGION_SOLVERS_NUM_INDEX_TYPES * LEGION_SOLVERS_NUM_INDEX_TYPES_2;

constexpr int LEGION_SOLVERS_MAX_DIM_0 = 1;
constexpr int LEGION_SOLVERS_MAX_DIM_1 =
    LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_0;
constexpr int LEGION_SOLVERS_MAX_DIM_2 =
    LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_1;
constexpr int LEGION_SOLVERS_MAX_DIM_3 =
    LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_2;

constexpr int LEGION_SOLVERS_TASK_BLOCK_SIZE = LEGION_SOLVERS_NUM_ENTRY_TYPES *
                                               LEGION_SOLVERS_MAX_DIM_3 *
                                               LEGION_SOLVERS_NUM_INDEX_TYPES_3;


template <
    Legion::TaskID BLOCK_ID,
    template <typename>
    typename TaskClass,
    typename T>
struct TaskT {

    static constexpr Legion::TaskID task_id =
        LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
        LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
        LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

    static std::string task_name() {
        return std::string(TaskClass<T>::task_base_name) + '_' +
               ToString<T>::value();
    }

    static void preregister(bool verbose = true) {
        preregister_task<
            typename TaskClass<T>::return_type,
            TaskClass<T>::task_body>(
            task_id,
            task_name(),
            Legion::Processor::LOC_PROC,
            TaskClass<T>::flags,
            verbose
        );
    }

}; // struct TaskT


template <
    Legion::TaskID BLOCK_ID,
    template <int, typename>
    typename TaskClass,
    int N,
    typename I>
struct TaskDI {

    static constexpr Legion::TaskID task_id =
        LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
        LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
        LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_INDEX_TYPE_INDEX<I> + (N - 1);

    static std::string task_name() {
        return std::string{TaskClass<N, I>::task_base_name} + '_' +
               std::to_string(N) + '_' + ToString<I>::value();
    }

    static void preregister(bool verbose = true) {
        preregister_task<
            typename TaskClass<N, I>::return_type,
            TaskClass<N, I>::task_body>(
            task_id,
            task_name(),
            Legion::Processor::LOC_PROC,
            TaskClass<N, I>::flags,
            verbose
        );
    }

}; // struct TaskDI


template <Legion::TaskID BLOCK_ID, template <int, typename> typename TaskClass>
struct TaskDI<BLOCK_ID, TaskClass, 0, void> {

    static constexpr Legion::TaskID task_id(int N, int I) {
        return LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
               LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
               LEGION_SOLVERS_MAX_DIM * I + (N - 1);
    }

    static Legion::TaskID task_id(Legion::IndexSpace index_space) {
        return task_id(
            index_space.get_dim(),
            index_space.get_type_tag() - 256 * index_space.get_dim()
        );
    }

}; // struct TaskDI<BLOCK_ID, TaskClass, 0>


template <
    Legion::TaskID BLOCK_ID,
    template <typename, int, typename>
    typename TaskClass,
    typename T,
    int N,
    typename I>
struct TaskTDI : HasCUDAVariantMixin {

    static constexpr Legion::TaskID task_id =
        LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
        LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
        LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM *
            LEGION_SOLVERS_INDEX_TYPE_INDEX<I> +
        LEGION_SOLVERS_NUM_ENTRY_TYPES * (N - 1) +
        LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

    static std::string task_name() {
        return std::string(TaskClass<T, N, I>::task_base_name) + '_' +
               ToString<T>::value() + '_' + std::to_string(N) + '_' +
               ToString<I>::value();
    }

    static void preregister(bool verbose) {
        preregister_task<
            typename TaskClass<T, N, I>::return_type,
            TaskClass<T, N, I>::task_body>(
            task_id,
            task_name(),
            Legion::Processor::LOC_PROC,
            TaskClass<T, N, I>::flags,
            verbose
        );

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
        if constexpr (TaskTDI::HasCUDAVariant<TaskClass<T, N, I>>::value) {
            preregister_task<
                typename TaskClass<T, N, I>::return_type,
                TaskClass<T, N, I>::cuda_task_body>(
                task_id,
                task_name(),
                Legion::Processor::TOC_PROC,
                TaskClass<T, N, I>::flags,
                verbose
            );
        }
#endif
    }

}; // struct TaskTDI


template <
    Legion::TaskID BLOCK_ID,
    template <typename, int, typename>
    typename TaskClass,
    typename T>
struct TaskTDI<BLOCK_ID, TaskClass, T, 0, void> {

    static constexpr Legion::TaskID task_id(int N, int I) {
        return LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
               LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
               LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM * I +
               LEGION_SOLVERS_NUM_ENTRY_TYPES * (N - 1) +
               LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;
    }

    static Legion::TaskID task_id(Legion::IndexSpace index_space) {
        return task_id(
            index_space.get_dim(),
            index_space.get_type_tag() - 256 * index_space.get_dim()
        );
    }

}; // struct TaskTDI


template <
    Legion::TaskID BLOCK_ID,
    template <typename, int, int, int, typename, typename, typename>
    typename TaskClass,
    typename T,
    int N1,
    int N2,
    int N3,
    typename I1,
    typename I2,
    typename I3>
struct TaskTDDDIII : HasCUDAVariantMixin {

    static constexpr Legion::TaskID task_id =
        LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
        LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
        LEGION_SOLVERS_NUM_INDEX_TYPES_3 * LEGION_SOLVERS_NUM_ENTRY_TYPES *
            LEGION_SOLVERS_MAX_DIM_2 * (N1 - 1) +
        LEGION_SOLVERS_NUM_INDEX_TYPES_3 * LEGION_SOLVERS_NUM_ENTRY_TYPES *
            LEGION_SOLVERS_MAX_DIM_1 * (N2 - 1) +
        LEGION_SOLVERS_NUM_INDEX_TYPES_3 * LEGION_SOLVERS_NUM_ENTRY_TYPES *
            LEGION_SOLVERS_MAX_DIM_0 * (N3 - 1) +
        LEGION_SOLVERS_NUM_INDEX_TYPES_2 * LEGION_SOLVERS_NUM_ENTRY_TYPES *
            LEGION_SOLVERS_INDEX_TYPE_INDEX<I1> +
        LEGION_SOLVERS_NUM_INDEX_TYPES_1 * LEGION_SOLVERS_NUM_ENTRY_TYPES *
            LEGION_SOLVERS_INDEX_TYPE_INDEX<I2> +
        LEGION_SOLVERS_NUM_INDEX_TYPES_0 * LEGION_SOLVERS_NUM_ENTRY_TYPES *
            LEGION_SOLVERS_INDEX_TYPE_INDEX<I3> +
        LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

    static std::string task_name() {
        return std::string(TaskClass<T, N1, N2, N3, I1, I2, I3>::task_base_name
               ) +
               '_' + ToString<T>::value() + '_' + std::to_string(N1) +
               std::to_string(N2) + std::to_string(N3) + '_' +
               ToString<I1>::value() + '_' + ToString<I2>::value() + '_' +
               ToString<I3>::value();
    }

    static void preregister(bool verbose) {
        preregister_task<
            typename TaskClass<T, N1, N2, N3, I1, I2, I3>::return_type,
            TaskClass<T, N1, N2, N3, I1, I2, I3>::task_body>(
            task_id,
            task_name(),
            Legion::Processor::LOC_PROC,
            TaskClass<T, N1, N2, N3, I1, I2, I3>::flags,
            verbose
        );

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
        if constexpr (TaskTDDDIII::HasCUDAVariant<
                          TaskClass<T, N1, N2, N3, I1, I2, I3>>::value) {
            preregister_task<
                typename TaskClass<T, N1, N2, N3, I1, I2, I3>::return_type,
                TaskClass<T, N1, N2, N3, I1, I2, I3>::cuda_task_body>(
                task_id,
                task_name(),
                Legion::Processor::TOC_PROC,
                TaskClass<T, N1, N2, N3, I1, I2, I3>::flags,
                verbose
            );
        }
#endif
    }

}; // struct TaskTDDDIII


template <
    Legion::TaskID BLOCK_ID,
    template <typename, int, int, int, typename, typename, typename>
    typename TaskClass,
    typename T>
struct TaskTDDDIII<BLOCK_ID, TaskClass, T, 0, 0, 0, void, void, void> {

    static constexpr Legion::TaskID
    task_id(int N1, int N2, int N3, int I1, int I2, int I3) {
        return LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
               LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
               LEGION_SOLVERS_NUM_INDEX_TYPES_3 *
                   LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM_2 *
                   (N1 - 1) +
               LEGION_SOLVERS_NUM_INDEX_TYPES_3 *
                   LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM_1 *
                   (N2 - 1) +
               LEGION_SOLVERS_NUM_INDEX_TYPES_3 *
                   LEGION_SOLVERS_NUM_ENTRY_TYPES * LEGION_SOLVERS_MAX_DIM_0 *
                   (N3 - 1) +
               LEGION_SOLVERS_NUM_INDEX_TYPES_2 *
                   LEGION_SOLVERS_NUM_ENTRY_TYPES * I1 +
               LEGION_SOLVERS_NUM_INDEX_TYPES_1 *
                   LEGION_SOLVERS_NUM_ENTRY_TYPES * I2 +
               LEGION_SOLVERS_NUM_INDEX_TYPES_0 *
                   LEGION_SOLVERS_NUM_ENTRY_TYPES * I3 +
               LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;
    }

    static Legion::TaskID task_id(
        Legion::IndexSpace kernel_space,
        Legion::IndexSpace domain_space,
        Legion::IndexSpace range_space
    ) {
        return task_id(
            kernel_space.get_dim(),
            domain_space.get_dim(),
            range_space.get_dim(),
            kernel_space.get_type_tag() - 256 * kernel_space.get_dim(),
            domain_space.get_type_tag() - 256 * domain_space.get_dim(),
            range_space.get_type_tag() - 256 * range_space.get_dim()
        );
    }

}; // struct TaskTDDDIII


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_TASK_BASE_CLASSES_HPP_INCLUDED

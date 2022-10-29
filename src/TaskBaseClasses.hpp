#ifndef LEGION_SOLVERS_TASK_BASE_CLASSES_HPP_INCLUDED
#define LEGION_SOLVERS_TASK_BASE_CLASSES_HPP_INCLUDED

#include <string> // for std::string

#include <legion.h> // for Legion::*

#include "LibraryOptions.hpp"           // for LEGION_SOLVERS_USE_*
#include "MetaprogrammingUtilities.hpp" // for TypeList, ListIndex, ...

namespace LegionSolvers {


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

    void>;


using LEGION_SOLVERS_SUPPORTED_ENTRY_TYPES = TypeList<
#ifdef LEGION_SOLVERS_USE_FLOAT
    float,
#endif // LEGION_SOLVERS_USE_FLOAT
#ifdef LEGION_SOLVERS_USE_DOUBLE
    double,
#endif // LEGION_SOLVERS_USE_DOUBLE
    void>;


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
        LEGION_SOLVERS_TASK_ID_ORIGIN +
        LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
        LEGION_SOLVERS_ENTRY_TYPE_INDEX<T>;

    static std::string task_name() {
        return (
            std::string(TaskClass<T>::task_base_name) + '_' +
            ToString<T>::value()
        );
    }

    static void preregister(bool verbose = true) {
        preregister_task<
            typename TaskClass<T>::return_type,
            TaskClass<T>::task_body>(
            task_id,
            task_name(),
            TaskClass<T>::flags,
            Legion::Processor::LOC_PROC,
            verbose
        );
    }

}; // struct TaskT


template <Legion::TaskID BLOCK_ID, template <int> typename TaskClass, int N>
struct TaskD {

    static constexpr Legion::TaskID task_id =
        LEGION_SOLVERS_TASK_ID_ORIGIN +
        LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + (N - 1);

    static std::string task_name() {
        return std::string{TaskClass<N>::task_base_name} + '_' +
               std::to_string(N);
    }

    static void preregister(bool verbose = true) {
        preregister_task<
            typename TaskClass<N>::return_type,
            TaskClass<N>::task_body>(
            task_id,
            task_name(),
            TaskClass<N>::flags,
            Legion::Processor::LOC_PROC,
            verbose
        );
    }

}; // struct TaskD


template <Legion::TaskID BLOCK_ID, template <int> typename TaskClass>
struct TaskD<BLOCK_ID, TaskClass, 0> {

    static constexpr Legion::TaskID task_id(int N) {
        return LEGION_SOLVERS_TASK_ID_ORIGIN +
               LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + (N - 1);
    }

}; // struct TaskD<BLOCK_ID, TaskClass, 0>

// Helper class to check if a class has a GPU task variant.
template <class T>
struct HasGPUVariantMixin {
    using __no = int8_t[1];
    using __yes = int8_t[2];
    struct HasGPUVariant {
        template <typename U>
        static __yes &test(decltype(&U::gpu_task_body));
        template <typename U>
        static __no &test(...);
        static const bool value = (sizeof(test<T>(0)) == sizeof(__yes));
    };
};

template <
    Legion::TaskID BLOCK_ID,
    template <typename, int, typename>
    typename TaskClass,
    typename T,
    int N,
    typename I>
struct TaskTDI : HasGPUVariantMixin<TaskTDI<BLOCK_ID, TaskClass, T, N, I>> {

    static constexpr Legion::TaskID task_id =
        LEGION_SOLVERS_TASK_ID_ORIGIN +
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
            TaskClass<T, N, I>::flags,
            Legion::Processor::LOC_PROC,
            verbose
        );

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
        if constexpr (TaskTDI::HasGPUVariant::value) {
            preregister_task<
                typename TaskClass<T, N, I>::return_type,
                TaskClass<T, N, I>::gpu_task_body>(
                task_id,
                task_name(),
                TaskClass<T, N, I>::flags,
                Legion::Processor::TOC_PROC,
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
        return LEGION_SOLVERS_TASK_ID_ORIGIN +
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
struct TaskTDDDIII
    : HasGPUVariantMixin<
          TaskTDDDIII<BLOCK_ID, TaskClass, T, N1, N2, N3, I1, I2, I3>> {

    static constexpr Legion::TaskID task_id =
        LEGION_SOLVERS_TASK_ID_ORIGIN +
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
            TaskClass<T, N1, N2, N3, I1, I2, I3>::flags,
            Legion::Processor::LOC_PROC,
            verbose
        );

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
        if constexpr (TaskTDDDIII::HasGPUVariant::value) {
            preregister_task<
                typename TaskClass<T, N1, N2, N3, I1, I2, I3>::return_type,
                TaskClass<T, N1, N2, N3, I1, I2, I3>::gpu_task_body>(
                task_id,
                task_name(),
                TaskClass<T, N1, N2, N3, I1, I2, I3>::flags,
                Legion::Processor::TOC_PROC,
                verbose
            );
            std::cout << "REGISTERING GPU TASK: " << TaskTDDDIII::task_name()
                      << std::endl;
        } else {
            std::cout << "NOT REGISTERING GPU TASK: "
                      << TaskTDDDIII::task_name() << std::endl;
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
        return LEGION_SOLVERS_TASK_ID_ORIGIN +
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

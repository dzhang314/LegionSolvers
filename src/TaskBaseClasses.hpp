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
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<float      >() { return "float"         ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<double     >() { return "double"        ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<long double>() { return "longdouble"    ; }
    template <> constexpr const char *LEGION_SOLVERS_TYPE_NAME<__float128 >() { return "float128"      ; }


    template <typename T>
    constexpr int LEGION_SOLVERS_TYPE_INDEX = ListIndex<LEGION_SOLVERS_SUPPORTED_TYPES, T>::value;
    constexpr int LEGION_SOLVERS_NUM_TYPES = ListLength<LEGION_SOLVERS_SUPPORTED_TYPES>::value;
    constexpr int LEGION_SOLVERS_MAX_DIM_0 = 1;
    constexpr int LEGION_SOLVERS_MAX_DIM_1 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_0;
    constexpr int LEGION_SOLVERS_MAX_DIM_2 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_1;
    constexpr int LEGION_SOLVERS_MAX_DIM_3 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_2;
    constexpr int LEGION_SOLVERS_TASK_BLOCK_SIZE = LEGION_SOLVERS_NUM_TYPES * LEGION_SOLVERS_MAX_DIM_3;


    template <Legion::TaskID BLOCK_ID,
              template <typename, int> typename TaskClass,
              typename T, int N>
    struct TaskTD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            LEGION_SOLVERS_NUM_TYPES * (N - 1) +
            LEGION_SOLVERS_TYPE_INDEX<T>;

        static std::string task_name() {
            return std::string{TaskClass<T, N>::task_base_name()} +
                   std::string{"_"} + LEGION_SOLVERS_TYPE_NAME<T>() +
                   std::string{"_"} + std::to_string(N);
        }

        static void preregister_cpu(bool verbose) {
            preregister_cpu_task<TaskClass<T, N>::task>(
                task_id, task_name(), TaskClass<T, N>::is_leaf(), verbose
            );
        }

        // static void preregister(bool verbose) {
        //     preregister_kokkos_task<
        //         typename TaskClass<T, N>::ReturnType,
        //         TaskClass<T, N>::template KokkosTaskBody
        //     >(task_id, task_name(), TaskClass<T, N>::is_leaf, verbose);
        // }

        // static void announce(const std::type_info &kokkos_execution_space_id,
        //                      Legion::Context ctx, Legion::Runtime *rt) {
        //     const Legion::Processor proc = rt->get_executing_processor(ctx);
        //     std::cout << "[LegionSolvers] Running task " << task_name()
        //               << " on processor " << proc
        //               << " (kind " << proc.kind()
        //               << ", " << kokkos_execution_space_id.name()
        //               << ")" << std::endl;
        // }

    }; // struct TaskTD


    template <Legion::TaskID BLOCK_ID,
              template <typename, int> typename TaskClass,
              typename T>
    struct TaskTD<BLOCK_ID, TaskClass, T, 0> {

        static constexpr Legion::TaskID task_id(int N) {
            return LEGION_SOLVERS_TASK_ID_ORIGIN +
                   LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                   LEGION_SOLVERS_NUM_TYPES * (N - 1) +
                   LEGION_SOLVERS_TYPE_INDEX<T>;
        }

    }; // struct TaskTD


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

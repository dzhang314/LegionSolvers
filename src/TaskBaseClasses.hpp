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
            return std::string{TaskClass<T, N>::task_base_name} +
                   '_' + LEGION_SOLVERS_TYPE_NAME<T>() +
                   '_' + std::to_string(N);
        }

        static void preregister_cpu(bool verbose) {
            preregister_cpu_task<
                typename TaskClass<T, N>::return_type,
                TaskClass<T, N>::task_body
            >(task_id, task_name(), TaskClass<T, N>::is_leaf, verbose);
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
            >(task_id, task_name(), TaskClass<T, N>::is_leaf, verbose);
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
            return LEGION_SOLVERS_TASK_ID_ORIGIN +
                   LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                   LEGION_SOLVERS_NUM_TYPES * (N - 1) +
                   LEGION_SOLVERS_TYPE_INDEX<T>;
        }

    }; // struct TaskTD


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

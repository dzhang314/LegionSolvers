#ifndef LEGION_SOLVERS_TASK_BASE_CLASSES_HPP
#define LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

#include <iostream>
#include <string>
#include <typeinfo>
#include <type_traits>

#include <legion.h>

#include <Kokkos_Core.hpp>

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
    constexpr int LEGION_SOLVERS_MAX_DIM_1 = LEGION_SOLVERS_MAX_DIM;
    constexpr int LEGION_SOLVERS_MAX_DIM_2 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_1;
    constexpr int LEGION_SOLVERS_MAX_DIM_3 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_2;
    constexpr int LEGION_SOLVERS_TASK_BLOCK_SIZE = LEGION_SOLVERS_NUM_TYPES * LEGION_SOLVERS_MAX_DIM_3;


    template <Legion::TaskID BLOCK_ID, typename T>
    struct TaskT {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskT


    template <Legion::TaskID BLOCK_ID, int DIM>
    struct TaskD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            (DIM - 1);

    }; // struct TaskD


    template <Legion::TaskID BLOCK_ID>
    struct TaskD<BLOCK_ID, 0> {

        static constexpr Legion::TaskID task_id(int DIM) {
            return LEGION_SOLVERS_TASK_ID_ORIGIN +
                   LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                   (DIM - 1);
        }

    }; // struct TaskD


    template <typename ReturnType, template <typename> typename TaskClass>
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
            if constexpr (std::is_void_v<ReturnType>) {
                Legion::Runtime::preregister_task_variant<
                    TaskClass<Kokkos::Serial>::body
                >(registrar, task_name.c_str());
            } else {
                Legion::Runtime::preregister_task_variant<
                    ReturnType, TaskClass<Kokkos::Serial>::body
                >(registrar, task_name.c_str());
            }
        }
        #endif

        #ifdef KOKKOS_ENABLE_OPENMP
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos OMP task "
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
            if constexpr (std::is_void_v<ReturnType>) {
                Legion::Runtime::preregister_task_variant<
                    TaskClass<Kokkos::OpenMP>::body
                >(registrar, task_name.c_str());
            } else {
                Legion::Runtime::preregister_task_variant<
                    ReturnType, TaskClass<Kokkos::OpenMP>::body
                >(registrar, task_name.c_str());
            }
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
            if constexpr (std::is_void_v<ReturnType>) {
                Legion::Runtime::preregister_task_variant<
                    TaskClass<Kokkos::Cuda>::body
                >(registrar, task_name.c_str());
            } else {
                Legion::Runtime::preregister_task_variant<
                    ReturnType, TaskClass<Kokkos::Cuda>::body
                >(registrar, task_name.c_str());
            }
        }
        #endif
    }


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

        static void preregister(bool verbose) {
            preregister_kokkos_task<
                typename TaskClass<T, N>::ReturnType,
                TaskClass<T, N>::template KokkosTaskBody
            >(task_id, task_name(), TaskClass<T, N>::is_leaf, verbose);
        }

        static void announce(const std::type_info &kokkos_execution_space_id,
                             Legion::Context ctx, Legion::Runtime *rt) {
            const Legion::Processor proc = rt->get_executing_processor(ctx);
            std::cout << "[LegionSolvers] Running task " << task_name()
                      << " on processor " << proc
                      << " (kind " << proc.kind()
                      << ", " << kokkos_execution_space_id.name()
                      << ")" << std::endl;
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


    template <Legion::TaskID BLOCK_ID, typename T, int DIM1, int DIM2, int DIM3>
    struct TaskTDDD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN +
            LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
            LEGION_SOLVERS_MAX_DIM_2 * LEGION_SOLVERS_NUM_TYPES * (DIM1 - 1) +
            LEGION_SOLVERS_MAX_DIM_1 * LEGION_SOLVERS_NUM_TYPES * (DIM2 - 1) +
            LEGION_SOLVERS_NUM_TYPES * (DIM3 - 1) +
            LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskTDDD


    template <Legion::TaskID BLOCK_ID, typename T>
    struct TaskTDDD<BLOCK_ID, T, 0, 0, 0> {

        static constexpr Legion::TaskID task_id(int DIM1, int DIM2, int DIM3) {
            return LEGION_SOLVERS_TASK_ID_ORIGIN +
                   LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                   LEGION_SOLVERS_MAX_DIM_2 * LEGION_SOLVERS_NUM_TYPES * (DIM1 - 1) +
                   LEGION_SOLVERS_MAX_DIM_1 * LEGION_SOLVERS_NUM_TYPES * (DIM2 - 1) +
                   LEGION_SOLVERS_NUM_TYPES * (DIM3 - 1) +
                   LEGION_SOLVERS_TYPE_INDEX<T>;
        }

    }; // struct TaskTDDD


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

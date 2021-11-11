#ifndef LEGION_SOLVERS_TASK_BASE_CLASSES_HPP
#define LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

#include <type_traits>

#include <Kokkos_Core.hpp>



namespace LegionSolvers {


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
              template <typename, int, int, int> typename TaskClass,
              typename T>
    struct TaskTDDD<BLOCK_ID, TaskClass, T, 0, 0, 0> {

        static constexpr Legion::TaskID task_id(int N1, int N2, int N3) {
            return (
                LEGION_SOLVERS_TASK_ID_ORIGIN +
                LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                LEGION_SOLVERS_MAX_DIM_2 * LEGION_SOLVERS_NUM_TYPES * (N1 - 1) +
                LEGION_SOLVERS_MAX_DIM_1 * LEGION_SOLVERS_NUM_TYPES * (N2 - 1) +
                LEGION_SOLVERS_NUM_TYPES * (N3 - 1) +
                LEGION_SOLVERS_TYPE_INDEX<T>
            );
        }

    }; // struct TaskTDDD


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


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_BASE_CLASSES_HPP

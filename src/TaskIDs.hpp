#ifndef LEGION_SOLVERS_TASK_IDS_HPP
#define LEGION_SOLVERS_TASK_IDS_HPP

#include <iostream>
#include <string>
#include <typeinfo>

#include <legion.h>

#include "MetaprogrammingUtilities.hpp"


namespace LegionSolvers {


    constexpr Legion::TaskID LEGION_SOLVERS_TASK_ID_ORIGIN = 1'000'000;
    constexpr int LEGION_SOLVERS_MAX_DIM = 3;
    using LEGION_SOLVERS_SUPPORTED_TYPES = TypeList<float, double, long double>;

    // clang-format off
    template <typename T> const char *LEGION_SOLVERS_TYPE_NAME   () { return typeid(T).name(); }
    template <> const char *LEGION_SOLVERS_TYPE_NAME<float      >() { return "float"         ; }
    template <> const char *LEGION_SOLVERS_TYPE_NAME<double     >() { return "double"        ; }
    template <> const char *LEGION_SOLVERS_TYPE_NAME<long double>() { return "longdouble"    ; }
    template <> const char *LEGION_SOLVERS_TYPE_NAME<__float128 >() { return "float128"      ; }
    // clang-format on


    template <typename T>
    constexpr int LEGION_SOLVERS_TYPE_INDEX = ListIndex<LEGION_SOLVERS_SUPPORTED_TYPES, T>::value;
    constexpr int LEGION_SOLVERS_NUM_TYPES = ListLength<LEGION_SOLVERS_SUPPORTED_TYPES>::value;
    constexpr int LEGION_SOLVERS_MAX_DIM_1 = LEGION_SOLVERS_MAX_DIM;
    constexpr int LEGION_SOLVERS_MAX_DIM_2 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_1;
    constexpr int LEGION_SOLVERS_MAX_DIM_3 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_2;
    constexpr int LEGION_SOLVERS_TASK_BLOCK_SIZE = LEGION_SOLVERS_NUM_TYPES * LEGION_SOLVERS_MAX_DIM_3;


    enum TaskBlockID : Legion::TaskID {

        ADDITION_TASK_BLOCK_ID,
        SUBTRACTION_TASK_BLOCK_ID,
        NEGATION_TASK_BLOCK_ID,
        MULTIPLICATION_TASK_BLOCK_ID,
        DIVISION_TASK_BLOCK_ID,
        CONSTANT_FILL_TASK_BLOCK_ID,
        COPY_TASK_BLOCK_ID,
        AXPY_TASK_BLOCK_ID,
        XPAY_TASK_BLOCK_ID,
        DOT_PRODUCT_TASK_BLOCK_ID,
        COO_MATVEC_TASK_BLOCK_ID,
        PRINT_VECTOR_TASK_BLOCK_ID,
        COO_RMATVEC_TASK_BLOCK_ID,
        COO_PRINT_TASK_BLOCK_ID,
        FILL_COO_NEGATIVE_LAPLACIAN_2D_TASK_BLOCK_ID,

    };


    template <TaskBlockID BLOCK_ID, typename T>
    struct TaskT {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskT


    template <TaskBlockID BLOCK_ID, int DIM>
    struct TaskD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + (DIM - 1);

    }; // struct TaskD


    template <TaskBlockID BLOCK_ID>
    struct TaskD<BLOCK_ID, 0> {

        static constexpr Legion::TaskID task_id(int DIM) {
            return LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + (DIM - 1);
        }

    }; // struct TaskD


    template <TaskBlockID BLOCK_ID, typename T, int DIM>
    struct TaskTD {

        static constexpr Legion::TaskID task_id = LEGION_SOLVERS_TASK_ID_ORIGIN +
                                                  LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                                                  LEGION_SOLVERS_NUM_TYPES * (DIM - 1) + LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskTD


    template <TaskBlockID BLOCK_ID, typename T>
    struct TaskTD<BLOCK_ID, T, 0> {

        static constexpr Legion::TaskID task_id(int DIM) {
            return LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                   LEGION_SOLVERS_NUM_TYPES * (DIM - 1) + LEGION_SOLVERS_TYPE_INDEX<T>;
        }

    }; // struct TaskTD


    template <TaskBlockID BLOCK_ID, typename T, int DIM1, int DIM2, int DIM3>
    struct TaskTDDD {

        static constexpr Legion::TaskID task_id = LEGION_SOLVERS_TASK_ID_ORIGIN +
                                                  LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                                                  LEGION_SOLVERS_MAX_DIM_2 * LEGION_SOLVERS_NUM_TYPES * (DIM1 - 1) +
                                                  LEGION_SOLVERS_MAX_DIM_1 * LEGION_SOLVERS_NUM_TYPES * (DIM2 - 1) +
                                                  LEGION_SOLVERS_NUM_TYPES * (DIM3 - 1) + LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskTDDD


    template <TaskBlockID BLOCK_ID, typename T>
    struct TaskTDDD<BLOCK_ID, T, 0, 0, 0> {

        static constexpr Legion::TaskID task_id(int DIM1, int DIM2, int DIM3) {
            return LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                   LEGION_SOLVERS_MAX_DIM_2 * LEGION_SOLVERS_NUM_TYPES * (DIM1 - 1) +
                   LEGION_SOLVERS_MAX_DIM_1 * LEGION_SOLVERS_NUM_TYPES * (DIM2 - 1) +
                   LEGION_SOLVERS_NUM_TYPES * (DIM3 - 1) + LEGION_SOLVERS_TYPE_INDEX<T>;
        }

    }; // struct TaskTDDD


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_IDS_HPP
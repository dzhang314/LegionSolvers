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
    using LEGION_SOLVERS_SUPPORTED_TYPES = TypeList<float, double, long double, __float128>;


    template <typename T>
    const char *LEGION_SOLVERS_TYPE_NAME() {
        return typeid(T).name();
    }

    template <>
    const char *LEGION_SOLVERS_TYPE_NAME<float>() {
        return "float";
    }

    template <>
    const char *LEGION_SOLVERS_TYPE_NAME<double>() {
        return "double";
    }

    template <>
    const char *LEGION_SOLVERS_TYPE_NAME<long double>() {
        return "longdouble";
    }

    template <>
    const char *LEGION_SOLVERS_TYPE_NAME<__float128>() {
        return "float128";
    }


    template <typename T>
    constexpr int LEGION_SOLVERS_TYPE_INDEX = ListIndex<LEGION_SOLVERS_SUPPORTED_TYPES, T>::value;
    constexpr int LEGION_SOLVERS_NUM_TYPES = ListLength<LEGION_SOLVERS_SUPPORTED_TYPES>::value;
    constexpr int LEGION_SOLVERS_MAX_DIM_1 = LEGION_SOLVERS_MAX_DIM;
    constexpr int LEGION_SOLVERS_MAX_DIM_2 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_1;
    constexpr int LEGION_SOLVERS_MAX_DIM_3 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_2;
    constexpr int LEGION_SOLVERS_TASK_BLOCK_SIZE = LEGION_SOLVERS_NUM_TYPES * LEGION_SOLVERS_MAX_DIM_3;


    enum TaskBlockID {

        ADDITION_TASK_BLOCK_ID = 0,
        SUBTRACTION_TASK_BLOCK_ID = 1,
        NEGATION_TASK_BLOCK_ID = 2,
        MULTIPLICATION_TASK_BLOCK_ID = 3,
        DIVISION_TASK_BLOCK_ID = 4,
        IS_NONEMPTY_TASK_BLOCK_ID = 5,
        CONSTANT_FILL_TASK_BLOCK_ID = 6,
        COPY_TASK_BLOCK_ID = 7,
        AXPY_TASK_BLOCK_ID = 8,
        XPAY_TASK_BLOCK_ID = 9,
        DOT_PRODUCT_TASK_BLOCK_ID = 10,
        COO_MATVEC_TASK_BLOCK_ID = 11,

    };


    template <void (*TASK_PTR)(
        const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id, const std::string &task_name) {
        std::cout << "Registering task " << task_name << " with ID " << task_id << "." << std::endl;
        Legion::TaskVariantRegistrar registrar(task_id, task_name.c_str());
        registrar.add_constraint(Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
        Legion::Runtime::preregister_task_variant<TASK_PTR>(registrar, task_name.c_str());
    }


    template <
        typename RETURN_T,
        RETURN_T (*TASK_PTR)(
            const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id, const std::string &task_name) {
        std::cout << "Registering task " << task_name << " with ID " << task_id << "." << std::endl;
        Legion::TaskVariantRegistrar registrar(task_id, task_name.c_str());
        registrar.add_constraint(Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
        Legion::Runtime::preregister_task_variant<RETURN_T, TASK_PTR>(registrar, task_name.c_str());
    }


    template <TaskBlockID BLOCK_ID, typename T>
    struct TaskT {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskT


    template <template <typename> typename TASK_CLASS>
    void preregister(const std::string &task_name) {
        preregister_cpu_task<float, TASK_CLASS<float>::task>(TASK_CLASS<float>::task_id,
                                                             (task_name + "_float").c_str());
        preregister_cpu_task<double, TASK_CLASS<double>::task>(TASK_CLASS<double>::task_id,
                                                               (task_name + "_double").c_str());
        preregister_cpu_task<long double, TASK_CLASS<long double>::task>(TASK_CLASS<long double>::task_id,
                                                                         (task_name + "_longdouble").c_str());
        preregister_cpu_task<__float128, TASK_CLASS<__float128>::task>(TASK_CLASS<__float128>::task_id,
                                                                       (task_name + "_float128").c_str());
    }


    template <TaskBlockID BLOCK_ID, int DIM>
    struct TaskD {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + (DIM - 1);

    }; // struct TaskD


    template <typename RETURN_T, template <int> typename TASK_CLASS>
    void preregister(const std::string &task_name) {
        preregister_cpu_task<RETURN_T, TASK_CLASS<1>::task>(TASK_CLASS<1>::task_id, (task_name + "_1").c_str());
        preregister_cpu_task<RETURN_T, TASK_CLASS<2>::task>(TASK_CLASS<2>::task_id, (task_name + "_2").c_str());
        preregister_cpu_task<RETURN_T, TASK_CLASS<3>::task>(TASK_CLASS<3>::task_id, (task_name + "_3").c_str());
    }


    template <TaskBlockID BLOCK_ID, typename T, int DIM>
    struct TaskTD {

        static constexpr Legion::TaskID task_id = LEGION_SOLVERS_TASK_ID_ORIGIN +
                                                  LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                                                  LEGION_SOLVERS_NUM_TYPES * (DIM - 1) + LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskTD


    template <TaskBlockID BLOCK_ID, typename T, int DIM1, int DIM2, int DIM3>
    struct TaskTDDD {

        static constexpr Legion::TaskID task_id = LEGION_SOLVERS_TASK_ID_ORIGIN +
                                                  LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID +
                                                  LEGION_SOLVERS_MAX_DIM_2 * LEGION_SOLVERS_NUM_TYPES * (DIM1 - 1) +
                                                  LEGION_SOLVERS_MAX_DIM_1 * LEGION_SOLVERS_NUM_TYPES * (DIM2 - 1) +
                                                  LEGION_SOLVERS_NUM_TYPES * (DIM3 - 1) + LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskTDDD


    template <template <typename, int...> typename TASK_CLASS, typename... TS>
    struct CartesianProductPreregistrationHelper;

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS>
    struct CartesianProductPreregistrationHelper<TASK_CLASS, T, IntList<MS...>, IntList<>> {
        static void execute() {
            preregister_cpu_task<TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                             TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                 std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                 ToString<IntList<MS...>>::value());
        }
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, typename... TS, int... MS>
    struct CartesianProductPreregistrationHelper<TASK_CLASS, TypeList<T, TS...>, IntList<MS...>, IntList<>> {
        static void execute() {
            preregister_cpu_task<TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                             TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                 std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                 ToString<IntList<MS...>>::value());
            CartesianProductPreregistrationHelper<TASK_CLASS, TypeList<TS...>, IntList<MS...>, IntList<>>::execute();
        }
    };

    template <template <typename, int...> typename TASK_CLASS, int... MS>
    struct CartesianProductPreregistrationHelper<TASK_CLASS, TypeList<>, IntList<MS...>, IntList<>> {
        static void execute() {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int... NS>
    struct CartesianProductPreregistrationHelper<TASK_CLASS, T, IntList<MS...>, IntList<0, NS...>> {
        static void execute() {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int N, int... NS>
    struct CartesianProductPreregistrationHelper<TASK_CLASS, T, IntList<MS...>, IntList<N, NS...>> {
        static void execute() {
            CartesianProductPreregistrationHelper<TASK_CLASS, T, IntList<MS...>, IntList<N - 1, NS...>>::execute();
            CartesianProductPreregistrationHelper<TASK_CLASS, T, IntList<MS..., N>, IntList<NS...>>::execute();
        }
    };


    template <template <typename, int...> typename TASK_CLASS, typename... TS>
    struct CartesianProductPreregistrationHelperRT;

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS>
    struct CartesianProductPreregistrationHelperRT<TASK_CLASS, T, IntList<MS...>, IntList<>> {
        static void execute() {
            preregister_cpu_task<T, TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                                TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                    ToString<IntList<MS...>>::value());
        }
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, typename... TS, int... MS>
    struct CartesianProductPreregistrationHelperRT<TASK_CLASS, TypeList<T, TS...>, IntList<MS...>, IntList<>> {
        static void execute() {
            preregister_cpu_task<T, TASK_CLASS<T, MS...>::task>(TASK_CLASS<T, MS...>::task_id,
                                                                TASK_CLASS<T, MS...>::task_name() + std::string{"_"} +
                                                                    std::string{LEGION_SOLVERS_TYPE_NAME<T>()} +
                                                                    ToString<IntList<MS...>>::value());
            CartesianProductPreregistrationHelperRT<TASK_CLASS, TypeList<TS...>, IntList<MS...>, IntList<>>::execute();
        }
    };

    template <template <typename, int...> typename TASK_CLASS, int... MS>
    struct CartesianProductPreregistrationHelperRT<TASK_CLASS, TypeList<>, IntList<MS...>, IntList<>> {
        static void execute() {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int... NS>
    struct CartesianProductPreregistrationHelperRT<TASK_CLASS, T, IntList<MS...>, IntList<0, NS...>> {
        static void execute() {}
    };

    template <template <typename, int...> typename TASK_CLASS, typename T, int... MS, int N, int... NS>
    struct CartesianProductPreregistrationHelperRT<TASK_CLASS, T, IntList<MS...>, IntList<N, NS...>> {
        static void execute() {
            CartesianProductPreregistrationHelperRT<TASK_CLASS, T, IntList<MS...>, IntList<N - 1, NS...>>::execute();
            CartesianProductPreregistrationHelperRT<TASK_CLASS, T, IntList<MS..., N>, IntList<NS...>>::execute();
        }
    };


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_IDS_HPP

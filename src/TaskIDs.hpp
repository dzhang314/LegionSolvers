#ifndef LEGION_SOLVERS_TASK_IDS_HPP
#define LEGION_SOLVERS_TASK_IDS_HPP

#include <iostream>

#include <legion.h>


namespace LegionSolvers {


    constexpr Legion::TaskID LEGION_SOLVERS_TASK_ID_ORIGIN = 1'000'000;


    constexpr int LEGION_SOLVERS_MAX_DIM = 3;
    constexpr int LEGION_SOLVERS_MAX_DIM_1 = LEGION_SOLVERS_MAX_DIM;
    constexpr int LEGION_SOLVERS_MAX_DIM_2 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_1;
    constexpr int LEGION_SOLVERS_MAX_DIM_3 = LEGION_SOLVERS_MAX_DIM * LEGION_SOLVERS_MAX_DIM_2;


    template <typename T>
    constexpr int LEGION_SOLVERS_TYPE_INDEX;

    template <>
    constexpr int LEGION_SOLVERS_TYPE_INDEX<float> = 0;

    template <>
    constexpr int LEGION_SOLVERS_TYPE_INDEX<double> = 1;


    constexpr int LEGION_SOLVERS_NUM_TYPES = 2;
    constexpr int LEGION_SOLVERS_TASK_BLOCK_SIZE = LEGION_SOLVERS_NUM_TYPES * LEGION_SOLVERS_MAX_DIM_3;


    enum TaskBlockID {

        ADDITION_TASK_BLOCK_ID = 0,
        SUBTRACTION_TASK_BLOCK_ID = 1,
        NEGATION_TASK_BLOCK_ID = 2,
        MULTIPLICATION_TASK_BLOCK_ID = 3,
        DIVISION_TASK_BLOCK_ID = 4,

    };


    template <void (*TaskPtr)(
        const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id, const char *task_name) {
        std::cout << "Registering task " << task_name << " with ID " << task_id << "." << std::endl;
        Legion::TaskVariantRegistrar registrar(task_id, task_name);
        registrar.add_constraint(Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
        Legion::Runtime::preregister_task_variant<TaskPtr>(registrar, task_name);
    }


    template <
        typename ReturnType,
        ReturnType (*TaskPtr)(
            const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id, const char *task_name) {
        std::cout << "Registering task " << task_name << " with ID " << task_id << "." << std::endl;
        Legion::TaskVariantRegistrar registrar(task_id, task_name);
        registrar.add_constraint(Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
        Legion::Runtime::preregister_task_variant<ReturnType, TaskPtr>(registrar, task_name);
    }


    template <TaskBlockID BLOCK_ID, typename T>
    struct TaskT {

        static constexpr Legion::TaskID task_id =
            LEGION_SOLVERS_TASK_ID_ORIGIN + LEGION_SOLVERS_TASK_BLOCK_SIZE * BLOCK_ID + LEGION_SOLVERS_TYPE_INDEX<T>;

    }; // struct TaskT


    template <template <typename> typename TaskStruct>
    void preregister(const std::string &task_name) {
        preregister_cpu_task<float, TaskStruct<float>::task>(TaskStruct<float>::task_id,
                                                             (task_name + "_float").c_str());
        preregister_cpu_task<double, TaskStruct<double>::task>(TaskStruct<double>::task_id,
                                                               (task_name + "_double").c_str());
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_IDS_HPP

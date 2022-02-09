#include <iostream>

#include <legion.h>

#include "DenseVector.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    TEST_FLOAT__1_SI_TASK_ID,
    TEST_FLOAT__1_UI_TASK_ID,
    TEST_FLOAT__1_LL_TASK_ID,
    TEST_FLOAT__2_SI_TASK_ID,
    TEST_FLOAT__2_UI_TASK_ID,
    TEST_FLOAT__2_LL_TASK_ID,
    TEST_FLOAT__3_SI_TASK_ID,
    TEST_FLOAT__3_UI_TASK_ID,
    TEST_FLOAT__3_LL_TASK_ID,
    TEST_DOUBLE_1_SI_TASK_ID,
    TEST_DOUBLE_1_UI_TASK_ID,
    TEST_DOUBLE_1_LL_TASK_ID,
    TEST_DOUBLE_2_SI_TASK_ID,
    TEST_DOUBLE_2_UI_TASK_ID,
    TEST_DOUBLE_2_LL_TASK_ID,
    TEST_DOUBLE_3_SI_TASK_ID,
    TEST_DOUBLE_3_UI_TASK_ID,
    TEST_DOUBLE_3_LL_TASK_ID,
    TOP_LEVEL_TASK_ID
};


template <int DIM, typename COORD_T>
Legion::Rect<DIM, COORD_T> create_rect(std::size_t size) {
    if constexpr (DIM == 1) {
        return {0, static_cast<COORD_T>(size) - 1};
    }
    if constexpr (DIM == 2) {
        return {{0, 0}, {
            static_cast<COORD_T>(std::floor(std::sqrt(size))) - 1,
            static_cast<COORD_T>(std::ceil(std::sqrt(size))) - 1
        }};
    }
    if constexpr (DIM == 3) {
        return {{0, 0, 0}, {
            static_cast<COORD_T>(std::floor(std::cbrt(size))) - 1,
            static_cast<COORD_T>(std::ceil(std::cbrt(size))) - 1,
            static_cast<COORD_T>(std::ceil(std::cbrt(size))) - 1,
        }};
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void test_task(const Legion::Task *,
               const std::vector<Legion::PhysicalRegion> &,
               Legion::Context ctx, Legion::Runtime *rt) {

    Legion::Rect<DIM, COORD_T> rect = create_rect<DIM, COORD_T>(100);
    Legion::IndexSpaceT<DIM, COORD_T> index_space = rt->create_index_space(ctx, rect);

    {
        LegionSolvers::DenseVector<ENTRY_T> u{ctx, rt, "u", index_space};
        LegionSolvers::DenseVector<ENTRY_T> v{ctx, rt, "v", u.get_index_space()};
        LegionSolvers::DenseVector<ENTRY_T> w{ctx, rt, "w", v.get_index_space()};

        u.random_fill();
        v.random_fill();
        w = u;

        w.axpy(1.0, v);
        v.xpay(-1.0, u);
        u.axpy(-0.5, v);
        u.axpy(-0.5, w);

        u.dot(u).assert_small();
    }

    rt->destroy_index_space(ctx, index_space);

}


void launch_task(Legion::Context ctx, Legion::Runtime *rt, Legion::TaskID id) {
    Legion::TaskLauncher launcher{id, Legion::UntypedBuffer{}};
    launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
    rt->execute_task(ctx, launcher);
}


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {
    // launch_task(ctx, rt, TEST_FLOAT__1_SI_TASK_ID);
    // launch_task(ctx, rt, TEST_FLOAT__1_UI_TASK_ID);
    launch_task(ctx, rt, TEST_FLOAT__1_LL_TASK_ID);
    // launch_task(ctx, rt, TEST_FLOAT__2_SI_TASK_ID);
    // launch_task(ctx, rt, TEST_FLOAT__2_UI_TASK_ID);
    launch_task(ctx, rt, TEST_FLOAT__2_LL_TASK_ID);
    // launch_task(ctx, rt, TEST_FLOAT__3_SI_TASK_ID);
    // launch_task(ctx, rt, TEST_FLOAT__3_UI_TASK_ID);
    launch_task(ctx, rt, TEST_FLOAT__3_LL_TASK_ID);
    // launch_task(ctx, rt, TEST_DOUBLE_1_SI_TASK_ID);
    // launch_task(ctx, rt, TEST_DOUBLE_1_UI_TASK_ID);
    launch_task(ctx, rt, TEST_DOUBLE_1_LL_TASK_ID);
    // launch_task(ctx, rt, TEST_DOUBLE_2_SI_TASK_ID);
    // launch_task(ctx, rt, TEST_DOUBLE_2_UI_TASK_ID);
    launch_task(ctx, rt, TEST_DOUBLE_2_LL_TASK_ID);
    // launch_task(ctx, rt, TEST_DOUBLE_3_SI_TASK_ID);
    // launch_task(ctx, rt, TEST_DOUBLE_3_UI_TASK_ID);
    launch_task(ctx, rt, TEST_DOUBLE_3_LL_TASK_ID);
}


int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<test_task<float , 1, int      >>(TEST_FLOAT__1_SI_TASK_ID, "test_float__1_si", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, unsigned >>(TEST_FLOAT__1_UI_TASK_ID, "test_float__1_ui", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, long long>>(TEST_FLOAT__1_LL_TASK_ID, "test_float__1_ll", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, int      >>(TEST_FLOAT__2_SI_TASK_ID, "test_float__2_si", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, unsigned >>(TEST_FLOAT__2_UI_TASK_ID, "test_float__2_ui", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, long long>>(TEST_FLOAT__2_LL_TASK_ID, "test_float__2_ll", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, int      >>(TEST_FLOAT__3_SI_TASK_ID, "test_float__3_si", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, unsigned >>(TEST_FLOAT__3_UI_TASK_ID, "test_float__3_ui", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, long long>>(TEST_FLOAT__3_LL_TASK_ID, "test_float__3_ll", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, int      >>(TEST_DOUBLE_1_SI_TASK_ID, "test_double_1_si", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, unsigned >>(TEST_DOUBLE_1_UI_TASK_ID, "test_double_1_ui", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, long long>>(TEST_DOUBLE_1_LL_TASK_ID, "test_double_1_ll", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, int      >>(TEST_DOUBLE_2_SI_TASK_ID, "test_double_2_si", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, unsigned >>(TEST_DOUBLE_2_UI_TASK_ID, "test_double_2_ui", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, long long>>(TEST_DOUBLE_2_LL_TASK_ID, "test_double_2_ll", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, int      >>(TEST_DOUBLE_3_SI_TASK_ID, "test_double_3_si", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, unsigned >>(TEST_DOUBLE_3_UI_TASK_ID, "test_double_3_ui", true, true, false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, long long>>(TEST_DOUBLE_3_LL_TASK_ID, "test_double_3_ll", true, true, false, false);
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", true, true, false, false
    );
    LegionSolvers::preregister_solver_tasks(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::set_top_level_task_mapper_id(
        LegionSolvers::LEGION_SOLVERS_MAPPER_ID);
    return Legion::Runtime::start(argc, argv);
}

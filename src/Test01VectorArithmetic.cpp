#include <cmath>
#include <vector>

#include <legion.h>

#include "DistributedVector.hpp"
#include "LegionUtilities.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    TEST_FLOAT__1D_1D_CT_CT_TASK_ID,
    TEST_FLOAT__1D_2D_CT_CT_TASK_ID,
    TEST_FLOAT__1D_3D_CT_CT_TASK_ID,
    TEST_FLOAT__2D_1D_CT_CT_TASK_ID,
    TEST_FLOAT__2D_2D_CT_CT_TASK_ID,
    TEST_FLOAT__2D_3D_CT_CT_TASK_ID,
    TEST_FLOAT__3D_1D_CT_CT_TASK_ID,
    TEST_FLOAT__3D_2D_CT_CT_TASK_ID,
    TEST_FLOAT__3D_3D_CT_CT_TASK_ID,
    TEST_DOUBLE_1D_1D_CT_CT_TASK_ID,
    TEST_DOUBLE_1D_2D_CT_CT_TASK_ID,
    TEST_DOUBLE_1D_3D_CT_CT_TASK_ID,
    TEST_DOUBLE_2D_1D_CT_CT_TASK_ID,
    TEST_DOUBLE_2D_2D_CT_CT_TASK_ID,
    TEST_DOUBLE_2D_3D_CT_CT_TASK_ID,
    TEST_DOUBLE_3D_1D_CT_CT_TASK_ID,
    TEST_DOUBLE_3D_2D_CT_CT_TASK_ID,
    TEST_DOUBLE_3D_3D_CT_CT_TASK_ID,
    TEST_FLOAT__1D_1D_CT_SI_TASK_ID,
    TEST_FLOAT__1D_2D_CT_SI_TASK_ID,
    TEST_FLOAT__1D_3D_CT_SI_TASK_ID,
    TEST_FLOAT__2D_1D_CT_SI_TASK_ID,
    TEST_FLOAT__2D_2D_CT_SI_TASK_ID,
    TEST_FLOAT__2D_3D_CT_SI_TASK_ID,
    TEST_FLOAT__3D_1D_CT_SI_TASK_ID,
    TEST_FLOAT__3D_2D_CT_SI_TASK_ID,
    TEST_FLOAT__3D_3D_CT_SI_TASK_ID,
    TEST_DOUBLE_1D_1D_CT_SI_TASK_ID,
    TEST_DOUBLE_1D_2D_CT_SI_TASK_ID,
    TEST_DOUBLE_1D_3D_CT_SI_TASK_ID,
    TEST_DOUBLE_2D_1D_CT_SI_TASK_ID,
    TEST_DOUBLE_2D_2D_CT_SI_TASK_ID,
    TEST_DOUBLE_2D_3D_CT_SI_TASK_ID,
    TEST_DOUBLE_3D_1D_CT_SI_TASK_ID,
    TEST_DOUBLE_3D_2D_CT_SI_TASK_ID,
    TEST_DOUBLE_3D_3D_CT_SI_TASK_ID,
    TEST_FLOAT__1D_1D_CT_UI_TASK_ID,
    TEST_FLOAT__1D_2D_CT_UI_TASK_ID,
    TEST_FLOAT__1D_3D_CT_UI_TASK_ID,
    TEST_FLOAT__2D_1D_CT_UI_TASK_ID,
    TEST_FLOAT__2D_2D_CT_UI_TASK_ID,
    TEST_FLOAT__2D_3D_CT_UI_TASK_ID,
    TEST_FLOAT__3D_1D_CT_UI_TASK_ID,
    TEST_FLOAT__3D_2D_CT_UI_TASK_ID,
    TEST_FLOAT__3D_3D_CT_UI_TASK_ID,
    TEST_DOUBLE_1D_1D_CT_UI_TASK_ID,
    TEST_DOUBLE_1D_2D_CT_UI_TASK_ID,
    TEST_DOUBLE_1D_3D_CT_UI_TASK_ID,
    TEST_DOUBLE_2D_1D_CT_UI_TASK_ID,
    TEST_DOUBLE_2D_2D_CT_UI_TASK_ID,
    TEST_DOUBLE_2D_3D_CT_UI_TASK_ID,
    TEST_DOUBLE_3D_1D_CT_UI_TASK_ID,
    TEST_DOUBLE_3D_2D_CT_UI_TASK_ID,
    TEST_DOUBLE_3D_3D_CT_UI_TASK_ID,
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


template <typename ENTRY_T, int VECTOR_DIM, int VECTOR_COLOR_DIM,
          typename VECTOR_COORD_T, typename VECTOR_COLOR_COORD_T>
void test_task(const Legion::Task *,
               const std::vector<Legion::PhysicalRegion> &,
               Legion::Context ctx, Legion::Runtime *rt) {

    using DistributedVector = LegionSolvers::DistributedVectorT<
        ENTRY_T, VECTOR_DIM, VECTOR_COLOR_DIM,
        VECTOR_COORD_T, VECTOR_COLOR_COORD_T
    >;

    using LegionSolvers::LEGION_SOLVERS_TYPE_NAME;

    const std::vector<VECTOR_COLOR_COORD_T> color_nums{1, 4, 10, 20};

    for (VECTOR_COORD_T vec_size = 500; vec_size <= 1000; vec_size += 100) {
        for (const VECTOR_COLOR_COORD_T num_colors : color_nums) {

            const auto index_rect =
                create_rect<VECTOR_DIM, VECTOR_COORD_T>(vec_size);
            const auto color_rect =
                create_rect<VECTOR_COLOR_DIM, VECTOR_COLOR_COORD_T>(num_colors);

            std::cout << "Running arithmetic test on "
                      << LEGION_SOLVERS_TYPE_NAME<ENTRY_T>()
                      << " vectors (" << VECTOR_DIM << "D, "
                      << LEGION_SOLVERS_TYPE_NAME<VECTOR_COORD_T>()
                      << ") of size " << index_rect.volume() << ", using "
                      << color_rect.volume() << " colors ("
                      << VECTOR_COLOR_DIM << "D, "
                      << LEGION_SOLVERS_TYPE_NAME<VECTOR_COLOR_COORD_T>()
                      << ")." << std::endl;

            const auto index_space = rt->create_index_space(ctx, index_rect);
            const auto color_space = rt->create_index_space(ctx, color_rect);

            DistributedVector u{"u", index_space, color_space, ctx, rt};
            DistributedVector v{"v", u.index_partition, ctx, rt};
            DistributedVector w{"w", u.index_partition, ctx, rt};

            u.random_fill();
            v.random_fill();
            w = u;

            w.axpy(1.0, v);
            v.xpay(-1.0, u);
            u.axpy(-0.5, v);
            u.axpy(-0.5, w);

            // u.print();
            u.dot(u).assert_small();

        }
    }
}


void launch_task(Legion::TaskID task_id,
                 Legion::Context ctx, Legion::Runtime *rt) {
    Legion::TaskLauncher launcher{task_id, Legion::TaskArgument{nullptr, 0}};
    rt->execute_task(ctx, launcher);
}


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {
    launch_task(TEST_FLOAT__1D_1D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_2D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_3D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_1D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_2D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_3D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_1D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_2D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_3D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_1D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_2D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_3D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_1D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_2D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_3D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_1D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_2D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_3D_CT_CT_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_1D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_2D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_3D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_1D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_2D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_3D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_1D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_2D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_3D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_1D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_2D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_3D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_1D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_2D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_3D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_1D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_2D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_3D_CT_SI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_1D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_2D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__1D_3D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_1D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_2D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__2D_3D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_1D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_2D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_FLOAT__3D_3D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_1D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_2D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_1D_3D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_1D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_2D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_2D_3D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_1D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_2D_CT_UI_TASK_ID, ctx, rt);
    launch_task(TEST_DOUBLE_3D_3D_CT_UI_TASK_ID, ctx, rt);
}


int main(int argc, char **argv) {
    LegionSolvers::preregister_solver_tasks(false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 1, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__1D_1D_CT_CT_TASK_ID, "test_float__1d_1d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 2, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__1D_2D_CT_CT_TASK_ID, "test_float__1d_2d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 3, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__1D_3D_CT_CT_TASK_ID, "test_float__1d_3d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 1, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__2D_1D_CT_CT_TASK_ID, "test_float__2d_1d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 2, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__2D_2D_CT_CT_TASK_ID, "test_float__2d_2d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 3, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__2D_3D_CT_CT_TASK_ID, "test_float__2d_3d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 1, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__3D_1D_CT_CT_TASK_ID, "test_float__3d_1d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 2, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__3D_2D_CT_CT_TASK_ID, "test_float__3d_2d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 3, Legion::coord_t, Legion::coord_t>>(TEST_FLOAT__3D_3D_CT_CT_TASK_ID, "test_float__3d_3d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 1, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_1D_1D_CT_CT_TASK_ID, "test_double_1d_1d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 2, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_1D_2D_CT_CT_TASK_ID, "test_double_1d_2d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 3, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_1D_3D_CT_CT_TASK_ID, "test_double_1d_3d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 1, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_2D_1D_CT_CT_TASK_ID, "test_double_2d_1d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 2, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_2D_2D_CT_CT_TASK_ID, "test_double_2d_2d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 3, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_2D_3D_CT_CT_TASK_ID, "test_double_2d_3d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 1, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_3D_1D_CT_CT_TASK_ID, "test_double_3d_1d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 2, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_3D_2D_CT_CT_TASK_ID, "test_double_3d_2d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 3, Legion::coord_t, Legion::coord_t>>(TEST_DOUBLE_3D_3D_CT_CT_TASK_ID, "test_double_3d_3d_ct_ct", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 1, Legion::coord_t, int            >>(TEST_FLOAT__1D_1D_CT_SI_TASK_ID, "test_float__1d_1d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 2, Legion::coord_t, int            >>(TEST_FLOAT__1D_2D_CT_SI_TASK_ID, "test_float__1d_2d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 3, Legion::coord_t, int            >>(TEST_FLOAT__1D_3D_CT_SI_TASK_ID, "test_float__1d_3d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 1, Legion::coord_t, int            >>(TEST_FLOAT__2D_1D_CT_SI_TASK_ID, "test_float__2d_1d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 2, Legion::coord_t, int            >>(TEST_FLOAT__2D_2D_CT_SI_TASK_ID, "test_float__2d_2d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 3, Legion::coord_t, int            >>(TEST_FLOAT__2D_3D_CT_SI_TASK_ID, "test_float__2d_3d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 1, Legion::coord_t, int            >>(TEST_FLOAT__3D_1D_CT_SI_TASK_ID, "test_float__3d_1d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 2, Legion::coord_t, int            >>(TEST_FLOAT__3D_2D_CT_SI_TASK_ID, "test_float__3d_2d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 3, Legion::coord_t, int            >>(TEST_FLOAT__3D_3D_CT_SI_TASK_ID, "test_float__3d_3d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 1, Legion::coord_t, int            >>(TEST_DOUBLE_1D_1D_CT_SI_TASK_ID, "test_double_1d_1d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 2, Legion::coord_t, int            >>(TEST_DOUBLE_1D_2D_CT_SI_TASK_ID, "test_double_1d_2d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 3, Legion::coord_t, int            >>(TEST_DOUBLE_1D_3D_CT_SI_TASK_ID, "test_double_1d_3d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 1, Legion::coord_t, int            >>(TEST_DOUBLE_2D_1D_CT_SI_TASK_ID, "test_double_2d_1d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 2, Legion::coord_t, int            >>(TEST_DOUBLE_2D_2D_CT_SI_TASK_ID, "test_double_2d_2d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 3, Legion::coord_t, int            >>(TEST_DOUBLE_2D_3D_CT_SI_TASK_ID, "test_double_2d_3d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 1, Legion::coord_t, int            >>(TEST_DOUBLE_3D_1D_CT_SI_TASK_ID, "test_double_3d_1d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 2, Legion::coord_t, int            >>(TEST_DOUBLE_3D_2D_CT_SI_TASK_ID, "test_double_3d_2d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 3, Legion::coord_t, int            >>(TEST_DOUBLE_3D_3D_CT_SI_TASK_ID, "test_double_3d_3d_ct_si", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 1, Legion::coord_t, unsigned       >>(TEST_FLOAT__1D_1D_CT_UI_TASK_ID, "test_float__1d_1d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 2, Legion::coord_t, unsigned       >>(TEST_FLOAT__1D_2D_CT_UI_TASK_ID, "test_float__1d_2d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 1, 3, Legion::coord_t, unsigned       >>(TEST_FLOAT__1D_3D_CT_UI_TASK_ID, "test_float__1d_3d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 1, Legion::coord_t, unsigned       >>(TEST_FLOAT__2D_1D_CT_UI_TASK_ID, "test_float__2d_1d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 2, Legion::coord_t, unsigned       >>(TEST_FLOAT__2D_2D_CT_UI_TASK_ID, "test_float__2d_2d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 2, 3, Legion::coord_t, unsigned       >>(TEST_FLOAT__2D_3D_CT_UI_TASK_ID, "test_float__2d_3d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 1, Legion::coord_t, unsigned       >>(TEST_FLOAT__3D_1D_CT_UI_TASK_ID, "test_float__3d_1d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 2, Legion::coord_t, unsigned       >>(TEST_FLOAT__3D_2D_CT_UI_TASK_ID, "test_float__3d_2d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<float , 3, 3, Legion::coord_t, unsigned       >>(TEST_FLOAT__3D_3D_CT_UI_TASK_ID, "test_float__3d_3d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 1, Legion::coord_t, unsigned       >>(TEST_DOUBLE_1D_1D_CT_UI_TASK_ID, "test_double_1d_1d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 2, Legion::coord_t, unsigned       >>(TEST_DOUBLE_1D_2D_CT_UI_TASK_ID, "test_double_1d_2d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 1, 3, Legion::coord_t, unsigned       >>(TEST_DOUBLE_1D_3D_CT_UI_TASK_ID, "test_double_1d_3d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 1, Legion::coord_t, unsigned       >>(TEST_DOUBLE_2D_1D_CT_UI_TASK_ID, "test_double_2d_1d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 2, Legion::coord_t, unsigned       >>(TEST_DOUBLE_2D_2D_CT_UI_TASK_ID, "test_double_2d_2d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 2, 3, Legion::coord_t, unsigned       >>(TEST_DOUBLE_2D_3D_CT_UI_TASK_ID, "test_double_2d_3d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 1, Legion::coord_t, unsigned       >>(TEST_DOUBLE_3D_1D_CT_UI_TASK_ID, "test_double_3d_1d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 2, Legion::coord_t, unsigned       >>(TEST_DOUBLE_3D_2D_CT_UI_TASK_ID, "test_double_3d_2d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<test_task<double, 3, 3, Legion::coord_t, unsigned       >>(TEST_DOUBLE_3D_3D_CT_UI_TASK_ID, "test_double_3d_3d_ct_ui", false, false);
    LegionSolvers::preregister_cpu_task<top_level_task>(TOP_LEVEL_TASK_ID, "top_level", false, false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}

#include <iostream>
#include <random>

#define LEGION_DISABLE_DEPRECATED_ENUMS

#include <legion.h>

#undef NDEBUG

#include "COOMatrix.hpp"
#include "CSRMatrix.hpp"
#include "ConjugateGradientSolver.hpp"
#include "ExampleSystems.hpp"
#include "Planner.hpp"
#include "TaskRegistration.hpp"

constexpr Legion::coord_t MATRIX_SIZE = 16;
constexpr Legion::coord_t NUM_NONZERO_ENTRIES = 3 * MATRIX_SIZE - 2;
constexpr Legion::coord_t NUM_INPUT_PARTITIONS = 4;
constexpr Legion::coord_t NUM_OUTPUT_PARTITIONS = 4;

enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID = 10,
    FILL_CSR_NEGATIVE_LAPLACIAN_1D_TASK_ID = 20,
    FILL_CSR_NEGATIVE_LAPLACIAN_1D_ROWPTR_TASK_ID = 21,
    PRINT_CSR_TASK_ID = 22,
    PRINT_CSR_ROWPTR_TASK_ID = 23,
};


enum CSRMatrixFieldIDs : Legion::FieldID {
    FID_COL,
    FID_ENTRY,
    FID_ROWPTR,
};


constexpr Legion::coord_t GRID_HEIGHT = 5;
constexpr Legion::coord_t GRID_WIDTH = 5;


enum COOMatrixFieldIDs : Legion::FieldID {
    FID_COO_I = 101,
    FID_COO_J = 102,
    FID_COO_ENTRY = 103,
};


enum VectorFieldIDs : Legion::FieldID {
    FID_VEC_ENTRY = 200,
};


void fill_csr_negative_laplacian_1d(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx,
                                    Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &csr_matrix = regions[0];

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::coord_t, 1> col_writer{csr_matrix, FID_COL};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 1> entry_writer{csr_matrix, FID_ENTRY};

    Legion::PointInDomainIterator<1> iter{csr_matrix};
    for (Legion::coord_t i = 0; i < MATRIX_SIZE; ++i) {

        if (i > 0) {
            entry_writer[*iter] = -1.0;
            col_writer[*iter] = i - 1;
            ++iter;
        }

        entry_writer[*iter] = +2.0;
        col_writer[*iter] = i;
        ++iter;

        if (i + 1 < MATRIX_SIZE) {
            entry_writer[*iter] = -1.0;
            col_writer[*iter] = i + 1;
            ++iter;
        }
    }
}

void fill_csr_negative_laplacian_1d_rowptr(const Legion::Task *task,
                                           const std::vector<Legion::PhysicalRegion> &regions,
                                           Legion::Context ctx,
                                           Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &csr_rowptr = regions[0];

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Rect<1>, 1> rowptr_writer{csr_rowptr, FID_ROWPTR};

    Legion::PointInDomainIterator<1> iter{csr_rowptr};
    for (Legion::coord_t i = 0; i < MATRIX_SIZE; ++i) {
        rowptr_writer[*iter] =
            Legion::Rect<1>{(i == 0) ? 0 : (3 * i - 1), (i + 1 == MATRIX_SIZE) ? (3 * MATRIX_SIZE - 3) : 3 * i + 1};
        ++iter;
    }
}


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx,
                    Legion::Runtime *rt) {

    const Legion::IndexSpaceT<1> kernel_is = rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_NONZERO_ENTRIES - 1});

    const Legion::LogicalRegionT<1> csr_matrix = LegionSolvers::create_region(
        kernel_is, {{sizeof(Legion::coord_t), FID_COL}, {sizeof(double), FID_ENTRY}}, ctx, rt);

    {
        Legion::TaskLauncher launcher{FILL_CSR_NEGATIVE_LAPLACIAN_1D_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{csr_matrix, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    const auto vec_is = rt->create_index_space(ctx, Legion::Rect<1>{0, MATRIX_SIZE - 1});
    const auto csr_rowptr = LegionSolvers::create_region(vec_is, {{sizeof(Legion::Rect<1>), FID_ROWPTR}}, ctx, rt);

    {
        Legion::TaskLauncher launcher{FILL_CSR_NEGATIVE_LAPLACIAN_1D_ROWPTR_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{csr_rowptr, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, csr_rowptr});
        launcher.add_field(0, FID_ROWPTR);
        rt->execute_task(ctx, launcher);
    }

    {
        Legion::TaskLauncher launcher{PRINT_CSR_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{csr_matrix, LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    {
        Legion::TaskLauncher launcher{PRINT_CSR_ROWPTR_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{csr_rowptr, LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_rowptr});
        launcher.add_field(0, FID_ROWPTR);
        rt->execute_task(ctx, launcher);
    }

    const Legion::IndexSpaceT<1> color_is = rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_OUTPUT_PARTITIONS - 1});
    const auto equal_partition = rt->create_equal_partition(ctx, vec_is, color_is);

    LegionSolvers::CSRMatrix<double, 1, 1> matrix_obj{
        csr_matrix, FID_COL, FID_ENTRY, csr_rowptr, FID_ROWPTR, equal_partition, equal_partition, ctx, rt};

    const auto kernel_range_partition = matrix_obj.kernel_partition_from_range_partition(equal_partition, ctx, rt);

    {
        Legion::IndexLauncher launcher{PRINT_CSR_TASK_ID, color_is, Legion::TaskArgument{nullptr, 0},
                                       Legion::ArgumentMap{}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{rt->get_logical_partition(csr_matrix, kernel_range_partition), 0,
                                      LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_index_space(ctx, launcher);
    }

    const auto kernel_domain_partition = matrix_obj.kernel_partition_from_domain_partition(equal_partition, ctx, rt);

    {
        Legion::IndexLauncher launcher{PRINT_CSR_TASK_ID, color_is, Legion::TaskArgument{nullptr, 0},
                                       Legion::ArgumentMap{}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{rt->get_logical_partition(csr_matrix, kernel_domain_partition), 0,
                                      LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_index_space(ctx, launcher);
    }
}


void print_csr_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx,
                    Legion::Runtime *rt) {
    assert(regions.size() == 1);
    const auto &csr_matrix = regions[0];
    const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::coord_t, 1> col_reader{csr_matrix, FID_COL};
    const Legion::FieldAccessor<LEGION_READ_ONLY, double, 1> entry_reader{csr_matrix, FID_ENTRY};
    std::cout << task->index_point << std::endl;
    for (Legion::PointInDomainIterator<1> iter{csr_matrix}; iter(); ++iter) {
        std::cout << task->index_point << *iter << ": " << col_reader[*iter] << ", " << entry_reader[*iter]
                  << std::endl;
    }
}

void print_csr_rowptr_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx,
                           Legion::Runtime *rt) {
    assert(regions.size() == 1);
    const auto &csr_rowptr = regions[0];
    const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::Rect<1>, 1> rowptr_reader{csr_rowptr, FID_ROWPTR};
    std::cout << task->index_point << std::endl;
    for (Legion::PointInDomainIterator<1> iter{csr_rowptr}; iter(); ++iter) {
        const Legion::Rect<1> rect = rowptr_reader[*iter];
        std::cout << task->index_point << *iter << ": " << rect << std::endl;
    }
}


int main(int argc, char **argv) {
    using namespace LegionSolvers;
    preregister_solver_tasks(false);

    preregister_cpu_task<top_level_task>(TOP_LEVEL_TASK_ID, "top_level");
    preregister_cpu_task<fill_csr_negative_laplacian_1d>(FILL_CSR_NEGATIVE_LAPLACIAN_1D_TASK_ID,
                                                         "fill_csr_negative_laplacian");
    preregister_cpu_task<fill_csr_negative_laplacian_1d_rowptr>(FILL_CSR_NEGATIVE_LAPLACIAN_1D_ROWPTR_TASK_ID,
                                                                "fill_csr_negative_laplacian_rowptr");
    preregister_cpu_task<print_csr_task>(PRINT_CSR_TASK_ID, "print_csr");
    preregister_cpu_task<print_csr_rowptr_task>(PRINT_CSR_ROWPTR_TASK_ID, "print_csr_rowptr");

    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}

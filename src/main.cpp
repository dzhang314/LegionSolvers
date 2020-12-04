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
#include "Tasks.hpp"

using LegionSolvers::ConjugateGradientSolver;
using LegionSolvers::COOMatrix;
using LegionSolvers::CSRMatrix;
using LegionSolvers::Planner;

constexpr Legion::coord_t MATRIX_SIZE = 16;
constexpr Legion::coord_t NUM_NONZERO_ENTRIES = 3 * MATRIX_SIZE - 2;
constexpr Legion::coord_t NUM_INPUT_PARTITIONS = 4;
constexpr Legion::coord_t NUM_OUTPUT_PARTITIONS = 4;

enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID = 10,
    FILL_COO_MATRIX_TASK_ID = 11,
    FILL_VECTOR_TASK_ID = 12,
    PRINT_TASK_ID = 13,
    PRINT_VEC_TASK_ID = 16,
    BOUNDARY_FILL_VECTOR_TASK_ID = 17,
    FILL_NEGATIVE_LAPLACIAN_2D_TASK_ID = 18,
    FILL_2D_PLANE_TASK_ID = 19,
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


Legion::LogicalRegionT<1> create_region(Legion::IndexSpaceT<1> index_space,
                                        const std::vector<std::pair<std::size_t, Legion::FieldID>> &fields,
                                        Legion::Context ctx,
                                        Legion::Runtime *rt) {
    Legion::FieldSpace field_space = rt->create_field_space(ctx);
    Legion::FieldAllocator allocator = rt->create_field_allocator(ctx, field_space);
    for (const auto [field_size, field_id] : fields) { allocator.allocate_field(field_size, field_id); }
    return rt->create_logical_region(ctx, index_space, field_space);
}


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

#if 0

    const Legion::IndexSpaceT<1> kernel_is = rt->create_index_space(
        ctx, Legion::Rect<1>{0, NUM_NONZERO_ENTRIES - 1});

    const Legion::LogicalRegionT<1> csr_matrix = create_region(
        kernel_is,
        {{sizeof(Legion::coord_t), FID_COL}, {sizeof(double), FID_ENTRY}}, ctx,
        rt);

    {
        Legion::TaskLauncher launcher{FILL_CSR_NEGATIVE_LAPLACIAN_1D_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            csr_matrix, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    const auto vec_is =
        rt->create_index_space(ctx, Legion::Rect<1>{0, MATRIX_SIZE - 1});
    const auto csr_rowptr =
        create_region(vec_is, {{sizeof(Legion::Rect<1>), FID_ROWPTR}}, ctx, rt);

    {
        Legion::TaskLauncher launcher{
            FILL_CSR_NEGATIVE_LAPLACIAN_1D_ROWPTR_TASK_ID,
            Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            csr_rowptr, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, csr_rowptr});
        launcher.add_field(0, FID_ROWPTR);
        rt->execute_task(ctx, launcher);
    }

    {
        Legion::TaskLauncher launcher{PRINT_CSR_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            csr_matrix, LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    {
        Legion::TaskLauncher launcher{PRINT_CSR_ROWPTR_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            csr_rowptr, LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_rowptr});
        launcher.add_field(0, FID_ROWPTR);
        rt->execute_task(ctx, launcher);
    }

    CSRMatrix<1, 1, double> matrix_obj{csr_matrix, FID_COL, FID_ENTRY,
                                       csr_rowptr, FID_ROWPTR};

    const Legion::IndexSpaceT<1> color_is = rt->create_index_space(
        ctx, Legion::Rect<1>{0, NUM_OUTPUT_PARTITIONS - 1});
    const auto equal_partition =
        rt->create_equal_partition(ctx, vec_is, color_is);

    const auto kernel_range_partition =
        matrix_obj.kernel_partition_from_range_partition(equal_partition, ctx,
                                                         rt);

    {
        Legion::IndexLauncher launcher{PRINT_CSR_TASK_ID, color_is,
                                       Legion::TaskArgument{nullptr, 0},
                                       Legion::ArgumentMap{}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            rt->get_logical_partition(csr_matrix, kernel_range_partition), 0,
            LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_index_space(ctx, launcher);
    }

    const auto kernel_domain_partition =
        matrix_obj.kernel_partition_from_domain_partition(equal_partition, ctx,
                                                          rt);

    {
        Legion::IndexLauncher launcher{PRINT_CSR_TASK_ID, color_is,
                                       Legion::TaskArgument{nullptr, 0},
                                       Legion::ArgumentMap{}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            rt->get_logical_partition(csr_matrix, kernel_domain_partition), 0,
            LEGION_READ_ONLY, LEGION_EXCLUSIVE, csr_matrix});
        launcher.add_field(0, FID_COL);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_index_space(ctx, launcher);
    }

#endif

    // Create matrix and two vector regions (input and output).
    const auto coo_matrix = create_region(
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_NONZERO_ENTRIES - 1}),
        {{sizeof(Legion::coord_t), FID_COO_I}, {sizeof(Legion::coord_t), FID_COO_J}, {sizeof(double), FID_COO_ENTRY}},
        ctx, rt);
    const Legion::IndexSpaceT<1> index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, MATRIX_SIZE - 1});
    const Legion::LogicalRegionT<1> input_vector =
        create_region(index_space, {{sizeof(double), FID_VEC_ENTRY}}, ctx, rt);
    const Legion::LogicalRegionT<1> output_vector =
        create_region(index_space, {{sizeof(double), FID_VEC_ENTRY}}, ctx, rt);

    // Partition input and output vectors.
    const Legion::IndexSpaceT<1> input_color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_INPUT_PARTITIONS - 1});
    const Legion::IndexSpaceT<1> output_color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_OUTPUT_PARTITIONS - 1});
    const Legion::IndexPartitionT<1> input_partition =
        Legion::IndexPartitionT<1>{rt->create_equal_partition(ctx, input_vector.get_index_space(), input_color_space)};
    const Legion::IndexPartitionT<1> output_partition = Legion::IndexPartitionT<1>{
        rt->create_equal_partition(ctx, output_vector.get_index_space(), output_color_space)};

    { // Fill matrix entries.
        Legion::TaskLauncher launcher{FILL_COO_MATRIX_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{coo_matrix, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, coo_matrix});
        launcher.add_field(0, FID_COO_I);
        launcher.add_field(0, FID_COO_J);
        launcher.add_field(0, FID_COO_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    { // Fill input vector entries.
        Legion::TaskLauncher launcher{FILL_VECTOR_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{input_vector, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, input_vector});
        launcher.add_field(0, FID_VEC_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    // Construct map of nonzero tiles.
    COOMatrix<1, 1, 1, double> matrix_obj{coo_matrix,      FID_COO_I,        FID_COO_J, FID_COO_ENTRY,
                                          input_partition, output_partition, ctx,       rt};

    // Launch matrix-vector multiplication tasks.
    matrix_obj.matvec(output_vector, FID_VEC_ENTRY, input_vector, FID_VEC_ENTRY, ctx, rt);

    { // Print output vector.
        Legion::TaskLauncher launcher{PRINT_VEC_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{output_vector, LEGION_READ_ONLY, LEGION_EXCLUSIVE, output_vector});
        launcher.add_field(0, FID_VEC_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    // Create another rhs vector and partition it.
    const Legion::IndexSpaceT<1> rhs_index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, MATRIX_SIZE - 1});
    const auto rhs_vector = create_region(rhs_index_space, {{sizeof(double), FID_VEC_ENTRY}}, ctx, rt);
    const auto rhs_partition = rt->create_equal_partition(ctx, rhs_vector.get_index_space(), input_color_space);

    { // Fill new rhs vector.
        Legion::TaskLauncher launcher{BOUNDARY_FILL_VECTOR_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{rhs_vector, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, rhs_vector});
        launcher.add_field(0, FID_VEC_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    Planner planner{};
    planner.add_rhs(output_vector, FID_VEC_ENTRY, output_partition);
    planner.add_rhs(rhs_vector, FID_VEC_ENTRY, rhs_partition);
    planner.add_coo_matrix<1, 1, 1, double>(0, 0, coo_matrix, FID_COO_I, FID_COO_J, FID_COO_ENTRY, ctx, rt);
    planner.add_coo_matrix<1, 1, 1, double>(1, 1, coo_matrix, FID_COO_I, FID_COO_J, FID_COO_ENTRY, ctx, rt);

    ConjugateGradientSolver solver{planner, ctx, rt};
    solver.set_max_iterations(17);
    solver.solve(ctx, rt);

    {
        Legion::TaskLauncher launcher{PRINT_VEC_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{solver.workspace[0], LEGION_READ_ONLY, LEGION_EXCLUSIVE, solver.workspace[0]});
        launcher.add_field(0, LegionSolvers::ConjugateGradientSolver::FID_CG_X);
        rt->execute_task(ctx, launcher);
    }

    {
        Legion::TaskLauncher launcher{PRINT_VEC_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{solver.workspace[1], LEGION_READ_ONLY, LEGION_EXCLUSIVE, solver.workspace[1]});
        launcher.add_field(0, LegionSolvers::ConjugateGradientSolver::FID_CG_X);
        rt->execute_task(ctx, launcher);
    }


#ifdef TEST_2D

    // Create matrix and two vector regions (input and output).
    const Legion::coord_t kernel_size = laplacian_2d_kernel_size(GRID_HEIGHT, GRID_WIDTH);
    const auto negative_laplacian = create_region(
        rt->create_index_space(ctx, Legion::Rect<1>{0, kernel_size - 1}),
        {{sizeof(Legion::Point<2>), FID_COO_I}, {sizeof(Legion::Point<2>), FID_COO_J}, {sizeof(double), FID_COO_ENTRY}},
        ctx, rt);

    // const Legion::IndexSpace index_space = rt->create_index_space(
    //     ctx, Legion::Rect<2>{{0, 0}, {GRID_HEIGHT - 1, GRID_WIDTH - 1}});
    const Legion::IndexSpace index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, MATRIX_SIZE - 1});

    const auto input_vector = create_region(index_space, {{sizeof(double), FID_VEC_ENTRY}}, ctx, rt);
    const auto output_vector = create_region(index_space, {{sizeof(double), FID_VEC_ENTRY}}, ctx, rt);

    // Partition input and output vectors.
    const auto input_color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_INPUT_PARTITIONS - 1});
    const auto output_color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_OUTPUT_PARTITIONS - 1});
    const auto input_partition = rt->create_equal_partition(ctx, index_space, input_color_space);
    const auto output_partition = rt->create_equal_partition(ctx, index_space, output_color_space);

    { // Fill matrix entries.
        Legion::TaskLauncher launcher{FILL_NEGATIVE_LAPLACIAN_2D_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{negative_laplacian, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, negative_laplacian});
        launcher.add_field(0, FID_COO_I);
        launcher.add_field(0, FID_COO_J);
        launcher.add_field(0, FID_COO_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    { // Fill input vector entries.
        Legion::TaskLauncher launcher{FILL_2D_PLANE_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{input_vector, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, input_vector});
        launcher.add_field(0, FID_VEC_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    // Construct map of nonzero tiles.
    COOMatrix matrix_obj{negative_laplacian, FID_COO_I,        FID_COO_J, FID_COO_ENTRY,
                         input_partition,    output_partition, ctx,       rt};

    // Launch matrix-vector multiplication tasks.
    matrix_obj.matvec(output_vector, FID_VEC_ENTRY, input_vector, FID_VEC_ENTRY, ctx, rt);

    Planner planner{};
    planner.add_rhs(output_vector, FID_VEC_ENTRY, output_partition);
    planner.add_coo_matrix(0, 0, negative_laplacian, FID_COO_I, FID_COO_J, FID_COO_ENTRY, ctx, rt);

    ConjugateGradientSolver solver{planner, ctx, rt};
    solver.set_max_iterations(300);
    solver.solve(ctx, rt);

#endif
}

void fill_coo_matrix_task(const Legion::Task *task,
                          const std::vector<Legion::PhysicalRegion> &regions,
                          Legion::Context ctx,
                          Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &coo_matrix = regions[0];

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::coord_t, 1> i_writer{coo_matrix, FID_COO_I};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::coord_t, 1> j_writer{coo_matrix, FID_COO_J};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 1> entry_writer{coo_matrix, FID_COO_ENTRY};

    if (NUM_NONZERO_ENTRIES == 3 * MATRIX_SIZE - 2) {
        Legion::PointInDomainIterator<1> iter{coo_matrix};
        for (Legion::coord_t i = 0; i < MATRIX_SIZE; ++i) {
            i_writer[*iter] = i;
            j_writer[*iter] = i;
            entry_writer[*iter] = 2.0;
            ++iter;
        }
        for (Legion::coord_t i = 0; i < MATRIX_SIZE - 1; ++i) {
            i_writer[*iter] = i + 1;
            j_writer[*iter] = i;
            entry_writer[*iter] = -1.0;
            ++iter;
            i_writer[*iter] = i;
            j_writer[*iter] = i + 1;
            entry_writer[*iter] = -1.0;
            ++iter;
        }
    } else {
        std::set<std::pair<Legion::coord_t, Legion::coord_t>> indices{};
        std::random_device rng{};
        std::uniform_int_distribution<Legion::coord_t> index_dist{0, MATRIX_SIZE - 1};
        std::uniform_real_distribution<double> entry_dist{0.0, 1.0};
        while (indices.size() < NUM_NONZERO_ENTRIES) { indices.emplace(index_dist(rng), index_dist(rng)); }
        for (Legion::PointInDomainIterator<1> iter{coo_matrix}; iter(); ++iter) {
            const auto [i, j] = *indices.begin();
            indices.erase(indices.begin());
            const double entry = entry_dist(rng);
            i_writer[*iter] = i;
            j_writer[*iter] = j;
            entry_writer[*iter] = entry;
            std::cout << *iter << ": " << i << ", " << j << ", " << entry << std::endl;
        }
    }
}

void fill_vector_task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context ctx,
                      Legion::Runtime *rt) {
    assert(regions.size() == 1);
    const auto &vector = regions[0];
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 1> entry_writer{vector, FID_VEC_ENTRY};
    std::random_device rng{};
    std::uniform_real_distribution<double> entry_dist{0.0, 1.0};
    for (Legion::PointInDomainIterator<1> iter{vector}; iter(); ++iter) {
        const double entry = entry_dist(rng);
        entry_writer[*iter] = entry;
        std::cout << *iter << ": " << entry << std::endl;
    }
}

void fill_2d_plane_task(const Legion::Task *task,
                        const std::vector<Legion::PhysicalRegion> &regions,
                        Legion::Context ctx,
                        Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &vector = regions[0];

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 2> entry_writer{vector, FID_VEC_ENTRY};

    for (Legion::PointInDomainIterator<2> iter{vector}; iter(); ++iter) {
        const auto [i, j] = *iter;
        entry_writer[*iter] = i + j;
    }
}

void boundary_fill_vector_task(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx,
                               Legion::Runtime *rt) {
    assert(regions.size() == 1);
    const auto &vector = regions[0];
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 1> entry_writer{vector, FID_VEC_ENTRY};
    for (Legion::PointInDomainIterator<1> iter{vector}; iter(); ++iter) {
        auto i = *iter;
        if (i[0] == 0) {
            entry_writer[*iter] = -1.0;
        } else if (i[0] == MATRIX_SIZE - 1) {
            entry_writer[*iter] = 2.0;
        } else {
            entry_writer[*iter] = 0.0;
        }
    }
}

void print_task(const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx,
                Legion::Runtime *rt) {
    assert(regions.size() == 1);
    const auto &coo_matrix = regions[0];
    const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::coord_t, 1> i_reader{coo_matrix, FID_COO_I};
    const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::coord_t, 1> j_reader{coo_matrix, FID_COO_J};
    const Legion::FieldAccessor<LEGION_READ_ONLY, double, 1> entry_reader{coo_matrix, FID_COO_ENTRY};
    std::cout << task->index_point << std::endl;
    for (Legion::PointInDomainIterator<1> iter{coo_matrix}; iter(); ++iter) {
        std::cout << task->index_point << *iter << ": " << i_reader[*iter] << ", " << j_reader[*iter] << ", "
                  << entry_reader[*iter] << std::endl;
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

void print_vec_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx,
                    Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &vector = regions[0];

    assert(task->regions.size() == 1);
    const auto &vector_req = task->regions[0];

    assert(vector_req.privilege_fields.size() == 1);
    const Legion::FieldID vector_fid = *vector_req.privilege_fields.begin();

    const Legion::FieldAccessor<LEGION_READ_ONLY, double, 1> entry_reader{vector, vector_fid};

    for (Legion::PointInDomainIterator<1> iter{vector}; iter(); ++iter) {
        std::cout << task->index_point << ' ' << *iter << ": " << entry_reader[*iter] << std::endl;
    }
}

int main(int argc, char **argv) {

    using LegionSolvers::preregister_cpu_task;

    preregister_solver_tasks<double, 1>();
    LegionSolvers::preregister<LegionSolvers::AdditionTask>("addition");
    LegionSolvers::preregister<LegionSolvers::SubtractionTask>("subtraction");
    LegionSolvers::preregister<LegionSolvers::NegationTask>("negation");
    LegionSolvers::preregister<LegionSolvers::MultiplicationTask>("multiplication");
    LegionSolvers::preregister<LegionSolvers::DivisionTask>("division");

    preregister_cpu_task<top_level_task>(TOP_LEVEL_TASK_ID, "top_level");
    preregister_cpu_task<fill_coo_matrix_task>(FILL_COO_MATRIX_TASK_ID, "fill_coo_matrix");
    preregister_cpu_task<fill_vector_task>(FILL_VECTOR_TASK_ID, "fill_vector");
    preregister_cpu_task<print_task>(PRINT_TASK_ID, "print");
    preregister_cpu_task<print_vec_task>(PRINT_VEC_TASK_ID, "print_vec_task");
    preregister_cpu_task<boundary_fill_vector_task>(BOUNDARY_FILL_VECTOR_TASK_ID, "boundary_fill");
    preregister_cpu_task<fill_negative_laplacian_2d_task>(FILL_NEGATIVE_LAPLACIAN_2D_TASK_ID,
                                                          "fill_negative_laplacian_2d");
    preregister_cpu_task<fill_2d_plane_task>(FILL_2D_PLANE_TASK_ID, "fill_2d_plane");
    preregister_cpu_task<fill_csr_negative_laplacian_1d>(FILL_CSR_NEGATIVE_LAPLACIAN_1D_TASK_ID,
                                                         "fill_csr_negative_laplacian");
    preregister_cpu_task<fill_csr_negative_laplacian_1d_rowptr>(FILL_CSR_NEGATIVE_LAPLACIAN_1D_ROWPTR_TASK_ID,
                                                                "fill_csr_negative_laplacian_rowptr");
    preregister_cpu_task<print_csr_task>(PRINT_CSR_TASK_ID, "print_csr");
    preregister_cpu_task<print_csr_rowptr_task>(PRINT_CSR_ROWPTR_TASK_ID, "print_csr_rowptr");

    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}

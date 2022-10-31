#include "CSRMatrixTasks.hpp"

using LegionSolvers::CSRMatvecTask;
using LegionSolvers::CSRRmatvecTask;


LEGION_SOLVERS_KDR_TEMPLATE
void LegionSolvers::CSRMatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(regions.size() == 4);
    const auto &output_vec = regions[0];
    const auto &csr_matrix = regions[1];
    const auto &rowptr_region = regions[2];
    const auto &input_vec = regions[3];

    assert(task->regions.size() == 4);
    const auto &output_req = task->regions[0];
    const auto &matrix_req = task->regions[1];
    const auto &rowptr_req = task->regions[2];
    const auto &input_req = task->regions[3];

    assert(output_req.privilege_fields.size() == 1);
    const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 2);

    assert(rowptr_req.privilege_fields.size() == 1);
    const Legion::FieldID fid_rowptr = *rowptr_req.privilege_fields.begin();

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::DomainT<RANGE_DIM, RANGE_COORD_T> output_domain =
        output_vec.get_bounds<RANGE_DIM, RANGE_COORD_T>();

    const AffineSumAccessor<ENTRY_T, RANGE_DIM, RANGE_COORD_T>
        output_writer(output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>);

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_reader(
        csr_matrix, args.fid_entry
    );

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_reader(csr_matrix, args.fid_col);

    const AffineReader<
        Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>,
        RANGE_DIM,
        RANGE_COORD_T>
        rowptr_reader(rowptr_region, fid_rowptr);

    const Legion::DomainT<DOMAIN_DIM, DOMAIN_COORD_T> input_domain =
        input_vec.get_bounds<DOMAIN_DIM, DOMAIN_COORD_T>();

    const AffineReader<ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T> input_reader(
        input_vec, input_fid
    );

    using KPointIter =
        Legion::PointInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>;
    using DPointIter = Legion::PointInDomainIterator<RANGE_DIM, RANGE_COORD_T>;

    for (KPointIter k_it(csr_matrix); k_it(); ++k_it) {

        const Legion::Point<KERNEL_DIM, KERNEL_COORD_T> kp = *k_it;

        Legion::Point<RANGE_DIM, RANGE_COORD_T> row;
        bool row_found = false;
        for (DPointIter rowptr_it(rowptr_region); rowptr_it(); ++rowptr_it) {
            const Legion::Point<RANGE_DIM, RANGE_COORD_T> cur_row = *rowptr_it;
            const Legion::Rect<KERNEL_DIM, KERNEL_COORD_T> rect =
                rowptr_reader[cur_row];
            if (rect.contains(kp)) {
                row_found = true;
                row = cur_row;
            }
        }
        assert(row_found);

        const Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T> col = col_reader[kp];
        if (input_domain.contains(col) && output_domain.contains(row)) {
            output_writer[row] <<= entry_reader[kp] * input_reader[col];
        }
    }
}


LEGION_SOLVERS_KDR_TEMPLATE
void CSRRmatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(false);
}


// clang-format off
template void CSRMatvecTask<float, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void CSRRmatvecTask<float, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void CSRMatvecTask<double, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void CSRRmatvecTask<double, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
// clang-format on

#include "COOMatrixTasks.hpp"

#include <iostream> // for std::cout

using LegionSolvers::COOMatvecTask;
using LegionSolvers::COOPrintTask;
using LegionSolvers::COORmatvecTask;


LEGION_SOLVERS_KDR_TEMPLATE
void COOMatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(regions.size() == 3);
    const auto &output_vec = regions[0];
    const auto &coo_matrix = regions[1];
    const auto &input_vec = regions[2];

    assert(task->regions.size() == 3);
    [[maybe_unused]] const auto &output_req = task->regions[0];
    [[maybe_unused]] const auto &matrix_req = task->regions[1];
    [[maybe_unused]] const auto &input_req = task->regions[2];

    assert(output_req.privilege_fields.size() == 1);
    const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 3);

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::DomainT<RANGE_DIM, RANGE_COORD_T> output_domain =
        output_vec.get_bounds<RANGE_DIM, RANGE_COORD_T>();

    const AffineSumAccessor<ENTRY_T, RANGE_DIM, RANGE_COORD_T>
        output_writer(output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>);

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_reader(
        coo_matrix, args.fid_entry
    );

    const AffineReader<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        row_reader(coo_matrix, args.fid_row);

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_reader(coo_matrix, args.fid_col);

    const Legion::DomainT<DOMAIN_DIM, DOMAIN_COORD_T> input_domain =
        input_vec.get_bounds<DOMAIN_DIM, DOMAIN_COORD_T>();

    const AffineReader<ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T> input_reader(
        input_vec, input_fid
    );

    using KPointIter =
        Legion::PointInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>;

    for (KPointIter k_it(coo_matrix); k_it(); ++k_it) {
        const Legion::Point<KERNEL_DIM, KERNEL_COORD_T> kp = *k_it;
        const Legion::Point<RANGE_DIM, RANGE_COORD_T> row = row_reader[kp];
        const Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T> col = col_reader[kp];
        if (input_domain.contains(col) && output_domain.contains(row)) {
            output_writer[row] <<= entry_reader[kp] * input_reader[col];
        }
    }
}


LEGION_SOLVERS_KDR_TEMPLATE
void COORmatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(false);
}


LEGION_SOLVERS_KDR_TEMPLATE
void COOPrintTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(regions.size() == 1);
    const auto &coo_matrix = regions[0];

    assert(task->regions.size() == 1);
    [[maybe_unused]] const auto &matrix_req = task->regions[0];

    assert(matrix_req.privilege_fields.size() == 3);

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_reader(
        coo_matrix, args.fid_entry
    );

    const AffineReader<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        row_reader(coo_matrix, args.fid_row);

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_reader(coo_matrix, args.fid_col);

    using KPointIter =
        Legion::PointInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>;

    for (KPointIter k_it(coo_matrix); k_it(); ++k_it) {
        const Legion::Point<KERNEL_DIM, KERNEL_COORD_T> kp = *k_it;
        const Legion::Point<RANGE_DIM, RANGE_COORD_T> row = row_reader[kp];
        const Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T> col = col_reader[kp];
        std::cout << kp << ": entry " << entry_reader[kp] << " at (" << row
                  << ", " << col << ")" << std::endl;
    }
}


// clang-format off
template void COOMatvecTask<float, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COORmatvecTask<float, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOPrintTask<float, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOPrintTask<float, 1, 2, 2, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOPrintTask<float, 1, 3, 3, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOMatvecTask<double, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COORmatvecTask<double, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOPrintTask<double, 1, 1, 1, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOPrintTask<double, 1, 2, 2, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOPrintTask<double, 1, 3, 3, long long, long long, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
// clang-format on

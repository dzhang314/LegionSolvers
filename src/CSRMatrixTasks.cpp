#include "CSRMatrixTasks.hpp"

using LegionSolvers::CSRMatvecTask;
using LegionSolvers::CSRRmatvecTask;

template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
void LegionSolvers::CSRMatvecTask<
    ENTRY_T,
    KERNEL_DIM,
    DOMAIN_DIM,
    RANGE_DIM,
    KERNEL_COORD_T,
    DOMAIN_COORD_T,
    RANGE_COORD_T>::
    task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {

    assert(regions.size() == 4);
    const auto &output_vec = regions[0];
    const auto &csr_matrix = regions[1];
    const auto &aux_region = regions[2];
    const auto &input_vec = regions[3];

    assert(task->regions.size() == 4);
    const auto &output_req = task->regions[0];
    const auto &matrix_req = task->regions[1];
    const auto &aux_req = task->regions[2];
    const auto &input_req = task->regions[3];

    assert(output_req.privilege_fields.size() == 1);
    const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 2);
    assert(task->arglen == 2 * sizeof(Legion::FieldID));
    const Legion::FieldID *argptr =
        reinterpret_cast<const Legion::FieldID *>(task->args);
    const Legion::FieldID fid_col = argptr[0];
    const Legion::FieldID fid_entry = argptr[1];

    assert(aux_req.privilege_fields.size() == 1);
    const Legion::FieldID fid_rowptr = *aux_req.privilege_fields.begin();

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_reader{csr_matrix, fid_col};

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_reader{
        csr_matrix, fid_entry};

    const AffineReader<
        Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>,
        RANGE_DIM,
        RANGE_COORD_T>
        rowptr_reader{aux_region, fid_rowptr};

    const AffineReader<ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T> input_reader{
        input_vec, input_fid};

    const AffineSumAccessor<ENTRY_T, RANGE_DIM, RANGE_COORD_T> output_writer{
        output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>};

    using KRectIter = Legion::RectInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>;
    using DRectIter = Legion::RectInDomainIterator<DOMAIN_DIM, DOMAIN_COORD_T>;
    using RRectIter = Legion::RectInDomainIterator<RANGE_DIM, RANGE_COORD_T>;

    for (KRectIter k_it(coo_matrix); k_it(); ++k_it) {
        const Legion::Rect<KERNEL_DIM, KERNEL_COORD_T> k_rect = *k_it;
        for (DRectIter d_it(input_vec); d_it(); ++d_it) {
            const Legion::Rect<DOMAIN_DIM, DOMAIN_COORD_T> d_rect = *d_it;
            for (RRectIter r_it(output_vec); r_it(); ++r_it) {
                const Legion::Rect<RANGE_DIM, RANGE_COORD_T> r_rect = *r_it;
                // TODO
            }
        }
    }
}

#include <tuple>

#include "COOMatrixTasks.hpp"


template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
template <typename ExecutionSpace>
void LegionSolvers::COOMatvecTask<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>::
KokkosTaskTemplate<ExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(regions.size() == 3);
    const auto &output_vec = regions[0];
    const auto &coo_matrix = regions[1];
    const auto &input_vec = regions[2];

    assert(task->regions.size() == 3);
    const auto &output_req = task->regions[0];
    const auto &matrix_req = task->regions[1];
    const auto &input_req = task->regions[2];

    assert(output_req.privilege_fields.size() == 1);
    const Legion::FieldID output_fid =
        *output_req.privilege_fields.begin();

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid =
        *input_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 3);
    assert(task->arglen == 3 * sizeof(Legion::FieldID));
    const Legion::FieldID *argptr =
        reinterpret_cast<const Legion::FieldID *>(task->args);
    const Legion::FieldID fid_i = argptr[0];
    const Legion::FieldID fid_j = argptr[1];
    const Legion::FieldID fid_entry = argptr[2];

    const Legion::FieldAccessor<
        LEGION_READ_ONLY,
        Legion::Point<RANGE_DIM>, KERNEL_DIM, Legion::coord_t,
        Realm::AffineAccessor<
            Legion::Point<RANGE_DIM>, KERNEL_DIM, Legion::coord_t
        >
    > i_reader{coo_matrix, fid_i};

    const Legion::FieldAccessor<
        LEGION_READ_ONLY,
        Legion::Point<DOMAIN_DIM>, KERNEL_DIM, Legion::coord_t,
        Realm::AffineAccessor<
            Legion::Point<DOMAIN_DIM>, KERNEL_DIM, Legion::coord_t
        >
    > j_reader{coo_matrix, fid_j};

    const Legion::FieldAccessor<
        LEGION_READ_ONLY, ENTRY_T, KERNEL_DIM, Legion::coord_t,
        Realm::AffineAccessor<ENTRY_T, KERNEL_DIM, Legion::coord_t>
    > entry_reader{coo_matrix, fid_entry};

    const Legion::FieldAccessor<
        LEGION_READ_ONLY, ENTRY_T, DOMAIN_DIM, Legion::coord_t,
        Realm::AffineAccessor<ENTRY_T, DOMAIN_DIM, Legion::coord_t>
    > input_reader{input_vec, input_fid};

    const Legion::ReductionAccessor<
        Legion::SumReduction<ENTRY_T>, false,
        RANGE_DIM, Legion::coord_t,
        Realm::AffineAccessor<ENTRY_T, RANGE_DIM, Legion::coord_t>
    > output_writer{
        output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>
    };

    for (Legion::RectInDomainIterator<KERNEL_DIM>
         kernel_iter{coo_matrix}; kernel_iter(); ++kernel_iter) {
        const Legion::Rect<KERNEL_DIM> kernel_rect = *kernel_iter;
        for (Legion::RectInDomainIterator<DOMAIN_DIM>
             domain_iter{input_vec}; domain_iter(); ++domain_iter) {
            const Legion::Rect<DOMAIN_DIM> domain_rect = *domain_iter;
            for (Legion::RectInDomainIterator<RANGE_DIM>
                 range_iter{output_vec}; range_iter(); ++range_iter) {
                const Legion::Rect<RANGE_DIM> range_rect = *range_iter;
                Kokkos::parallel_for(
                    KokkosRangeFactory<ExecutionSpace, KERNEL_DIM>::create(
                        kernel_rect, ctx, rt
                    ),
                    KokkosCOOMatvecFunctor<
                        ExecutionSpace, ENTRY_T,
                        KERNEL_DIM, DOMAIN_DIM, RANGE_DIM
                    >{
                        domain_rect, range_rect,
                        i_reader.accessor, j_reader.accessor,
                        entry_reader.accessor,
                        input_reader.accessor, output_writer
                    }
                );
            }
        }
    }
}


[[maybe_unused]]
static constexpr auto instantiations = std::make_tuple(
    LegionSolvers::COOMatvecTask<float, 1, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 1, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 2, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<float, 3, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 1, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 2, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::COOMatvecTask<double, 3, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body
);

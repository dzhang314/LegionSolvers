#include "COOMatrixTasks.hpp"
#include "LegionUtilities.hpp"


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

    // TODO
    using KERNEL_COORD_T = Legion::coord_t;
    using DOMAIN_COORD_T = Legion::coord_t;
    using RANGE_COORD_T = Legion::coord_t;

    const AffineReader<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>, KERNEL_DIM, KERNEL_COORD_T
    > i_reader{coo_matrix, fid_i};

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>, KERNEL_DIM, KERNEL_COORD_T
    > j_reader{coo_matrix, fid_j};

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T>
    entry_reader{coo_matrix, fid_entry};

    const AffineReader<ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T>
    input_reader{input_vec, input_fid};

    const AffineSumAccessor<ENTRY_T, RANGE_DIM, RANGE_COORD_T>
    output_writer{output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>};

    for (const auto &kernel_rect :
         Rects<KERNEL_DIM, KERNEL_COORD_T>{coo_matrix}) {
        for (const auto &domain_rect :
             Rects<DOMAIN_DIM, DOMAIN_COORD_T>{input_vec}) {
            for (const auto &range_rect :
                 Rects<RANGE_DIM, RANGE_COORD_T>{output_vec}) {
                Kokkos::parallel_for(
                    KokkosRangeFactory<ExecutionSpace, KERNEL_DIM>::create(
                        kernel_rect, ctx, rt
                    ),
                    KokkosCOOMatvecFunctor<
                        ExecutionSpace, ENTRY_T,
                        KERNEL_DIM, DOMAIN_DIM, RANGE_DIM,
                        KERNEL_COORD_T, DOMAIN_COORD_T, RANGE_COORD_T
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


template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
void LegionSolvers::COOPrintTask<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>::
task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(regions.size() == 1);
    const auto &coo_matrix = regions[0];

    assert(task->regions.size() == 1);
    const auto &matrix_req = task->regions[0];

    assert(matrix_req.privilege_fields.size() == 3);
    assert(task->arglen == 3 * sizeof(Legion::FieldID));
    const Legion::FieldID *argptr =
        reinterpret_cast<const Legion::FieldID *>(task->args);
    const Legion::FieldID fid_i = argptr[0];
    const Legion::FieldID fid_j = argptr[1];
    const Legion::FieldID fid_entry = argptr[2];

    // TODO
    using KERNEL_COORD_T = Legion::coord_t;
    using DOMAIN_COORD_T = Legion::coord_t;
    using RANGE_COORD_T = Legion::coord_t;

    const AffineReader<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>, KERNEL_DIM, KERNEL_COORD_T
    > i_reader{coo_matrix, fid_i};

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>, KERNEL_DIM, KERNEL_COORD_T
    > j_reader{coo_matrix, fid_j};

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T>
    entry_reader{coo_matrix, fid_entry};

    std::cout << "[LegionSolvers] Printing COO matrix:" << std::endl;
    for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix};
            iter(); ++iter) {
        const Legion::Point<RANGE_DIM, RANGE_COORD_T> i{i_reader[*iter]};
        const Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T> j{j_reader[*iter]};
        const ENTRY_T entry = entry_reader[*iter];
        std::cout << "[LegionSolvers]     ";
        if (task->is_index_space) {
            std::cout << task->index_point << ' ';
        }
        std::cout << *iter << ": (" << i << ", " << j << "), "
                    << entry << std::endl;
    }
}


template void LegionSolvers::COOMatvecTask<float , 1, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 1, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 2, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<float , 3, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 1, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 2, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 1, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 2, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 1>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 2>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOMatvecTask<double, 3, 3, 3>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);


template void LegionSolvers::COOPrintTask<float , 1, 1, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 1, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 1, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 2, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 2, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 2, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 3, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 3, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 1, 3, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 1, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 1, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 1, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 2, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 2, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 2, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 3, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 3, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 2, 3, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 1, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 1, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 1, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 2, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 2, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 2, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 3, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 3, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<float , 3, 3, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 1, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 1, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 1, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 2, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 2, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 2, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 3, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 3, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 1, 3, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 1, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 1, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 1, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 2, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 2, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 2, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 3, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 3, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 2, 3, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 1, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 1, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 1, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 2, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 2, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 2, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 3, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 3, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::COOPrintTask<double, 3, 3, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

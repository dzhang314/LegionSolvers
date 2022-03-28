#include "ExampleSystems.hpp"

#include <iostream>


template <typename T>
template <typename KokkosExecutionSpace>
void LegionSolvers::FillCOONegativeLaplacian1DTask<T>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    std::cout << "[LegionSolvers] Constructing COO 1D Laplacian matrix..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->regions.size() == 1);
    const auto &matrix_req = task->regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const AffineWriter<Legion::Point<1>, 1, Legion::coord_t>
    i_writer{matrix, args.fid_i};
    const AffineWriter<Legion::Point<1>, 1, Legion::coord_t>
    j_writer{matrix, args.fid_j};
    const AffineWriter<T, 1, Legion::coord_t>
    entry_writer{matrix, args.fid_entry};

    const Legion::Domain matrix_domain = rt->get_index_space_domain(
        ctx, matrix_req.region.get_index_space()
    );

    for (Legion::RectInDomainIterator<1> it{matrix_domain}; it(); ++it) {
        const Legion::Rect<1> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, 1, Legion::coord_t>::create(ctx, rt,
                rect
            ),
            KokkosFillCOONegativeLaplacian1DFunctor<KokkosExecutionSpace, T>{
                i_writer.accessor, j_writer.accessor, entry_writer.accessor
            }
        );
    }

    std::cout << "[LegionSolvers] Finished constructing COO 1D Laplacian."
              << std::endl;
}


template <typename T>
template <typename KokkosExecutionSpace>
void LegionSolvers::FillCSRNegativeLaplacian1DTask<T>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    std::cout << "[LegionSolvers] Constructing CSR 1D Laplacian matrix..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->regions.size() == 1);
    const auto &matrix_req = task->regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const AffineWriter<Legion::Point<1>, 1, Legion::coord_t>
    col_writer{matrix, args.fid_col};
    const AffineWriter<T, 1, Legion::coord_t>
    entry_writer{matrix, args.fid_entry};

    const Legion::Domain matrix_domain = rt->get_index_space_domain(
        ctx, matrix_req.region.get_index_space()
    );

    for (Legion::RectInDomainIterator<1> it{matrix_domain}; it(); ++it) {
        const Legion::Rect<1> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, 1, Legion::coord_t>::create(ctx, rt,
                rect
            ),
            KokkosFillCSRNegativeLaplacian1DFunctor<KokkosExecutionSpace, T>{
                col_writer.accessor, entry_writer.accessor
            }
        );
    }

    std::cout << "[LegionSolvers] Finished constructing CSR 1D Laplacian."
              << std::endl;
}


template <typename T>
template <typename KokkosExecutionSpace>
void LegionSolvers::FillCSRNegativeLaplacian1DRowptrTask<T>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    std::cout << "[LegionSolvers] Constructing CSR 1D Laplacian row pointer array..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &rowptr = regions[0];

    assert(task->regions.size() == 1);
    const auto &rowptr_req = task->regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const AffineWriter<Legion::Rect<1>, 1, Legion::coord_t>
    rowptr_writer{rowptr, args.fid_rowptr};

    const Legion::Domain rowptr_domain = rt->get_index_space_domain(
        ctx, rowptr_req.region.get_index_space()
    );

    for (Legion::RectInDomainIterator<1> it{rowptr_domain}; it(); ++it) {
        const Legion::Rect<1> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, 1, Legion::coord_t>::create(ctx, rt,
                rect
            ),
            KokkosFillCSRNegativeLaplacian1DRowptrFunctor<KokkosExecutionSpace, T>{
                rowptr_writer.accessor, args.grid_length
            }
        );
    }

    std::cout << "[LegionSolvers] Finished constructing CSR 1D Laplacian row pointers."
              << std::endl;
}


template <typename T>
void LegionSolvers::FillCOONegativeLaplacian2DTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    std::cout << "[LegionSolvers] Constructing COO 2D Laplacian matrix..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1>
    i_writer{matrix, args.fid_i};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1>
    j_writer{matrix, args.fid_j};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, 1>
    entry_writer{matrix, args.fid_entry};

    Legion::PointInDomainIterator<1> iter{matrix};
    for (Legion::coord_t i = 0; i < args.grid_height; ++i) {
        for (Legion::coord_t j = 0; j < args.grid_width; ++j) {
            i_writer[*iter] = Legion::Point<2>{i, j};
            j_writer[*iter] = Legion::Point<2>{i, j};
            entry_writer[*iter] = static_cast<T>(4.0);
            ++iter;
            if (i > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i - 1, j};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (j > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j - 1};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (i + 1 < args.grid_height) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i + 1, j};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (j + 1 < args.grid_width) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j + 1};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
        }
    }
    std::cout << "[LegionSolvers] Finished constructing COO 2D Laplacian."
              << std::endl;
}


#ifdef KOKKOS_ENABLE_SERIAL
template void LegionSolvers::FillCOONegativeLaplacian1DTask<float >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCOONegativeLaplacian1DTask<double>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DTask<float >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DTask<double>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DRowptrTask<float >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DRowptrTask<double>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
#endif // KOKKOS_ENABLE_SERIAL


#ifdef KOKKOS_ENABLE_OPENMP
template void LegionSolvers::FillCOONegativeLaplacian1DTask<float >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCOONegativeLaplacian1DTask<double>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DTask<float >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DTask<double>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DRowptrTask<float >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DRowptrTask<double>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
#endif // KOKKOS_ENABLE_OPENMP


#ifdef KOKKOS_ENABLE_CUDA
template void LegionSolvers::FillCOONegativeLaplacian1DTask<float >::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCOONegativeLaplacian1DTask<double>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DTask<float >::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DTask<double>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DRowptrTask<float >::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCSRNegativeLaplacian1DRowptrTask<double>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
#endif // KOKKOS_ENABLE_CUDA


template void LegionSolvers::FillCOONegativeLaplacian2DTask<float >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCOONegativeLaplacian2DTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

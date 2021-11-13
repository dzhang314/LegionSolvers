#include <tuple>

#include "COOMatrixTasks.hpp"


class RectIteratorSentinel {};


template <int DIM, typename COORD_T>
class RectIterator {

    Legion::RectInDomainIterator<DIM, COORD_T> iterator;

public:

    explicit RectIterator(
        const Legion::PhysicalRegion &region
    ) : iterator(region) {}

    Legion::Rect<DIM, COORD_T> operator*() {
        return *iterator;
    }

    RectIterator &operator++() {
        ++iterator;
        return *this;
    }

    bool operator!=(RectIteratorSentinel) {
        return iterator();
    }

};


template <int DIM, typename COORD_T>
class Rects {

    const Legion::PhysicalRegion &region;

public:

    explicit constexpr Rects(
        const Legion::PhysicalRegion &region
    ) noexcept : region(region) {}

    RectIterator<DIM, COORD_T> begin() {
        return RectIterator<DIM, COORD_T>{region};
    }

    constexpr RectIteratorSentinel end() noexcept {
        return RectIteratorSentinel{};
    }

};


constexpr bool LEGION_SOLVERS_CHECK_BOUNDS = true;


template <typename FIELD_TYPE, int DIM, typename COORD_T>
using AffineReader = Legion::FieldAccessor<
    LEGION_READ_ONLY, FIELD_TYPE, DIM, COORD_T,
    Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
    LEGION_SOLVERS_CHECK_BOUNDS
>;


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

    using KERNEL_COORD_T = Legion::coord_t;
    using DOMAIN_COORD_T = Legion::coord_t;
    using RANGE_COORD_T = Legion::coord_t;

    const AffineReader<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>, KERNEL_DIM, KERNEL_COORD_T
    > i_reader{coo_matrix, fid_i};

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>, KERNEL_DIM, KERNEL_COORD_T
    > j_reader{coo_matrix, fid_j};

    const AffineReader<
        ENTRY_T, KERNEL_DIM, KERNEL_COORD_T
    > entry_reader{coo_matrix, fid_entry};

    const AffineReader<
        ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T
    > input_reader{input_vec, input_fid};

    const Legion::ReductionAccessor<
        Legion::SumReduction<ENTRY_T>, false,
        RANGE_DIM, Legion::coord_t,
        Realm::AffineAccessor<ENTRY_T, RANGE_DIM, Legion::coord_t>
    > output_writer{
        output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>
    };

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

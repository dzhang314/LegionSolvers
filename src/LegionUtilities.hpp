#ifndef LEGION_SOLVERS_LEGION_UTILITIES_HPP
#define LEGION_SOLVERS_LEGION_UTILITIES_HPP

#include <cassert>
#include <cstddef>     // for std::size_t
#include <type_traits> // for std::is_void_v
#include <utility>     // for std::pair
#include <vector>      // for std::vector

#include <Kokkos_Core.hpp>
#include <legion.h>

#include "LibraryOptions.hpp"


namespace LegionSolvers {


    inline Legion::FieldSpace create_field_space(
        const std::vector<std::size_t> &field_sizes,
        const std::vector<Legion::FieldID> &field_ids,
        Legion::Context ctx, Legion::Runtime *rt
    ) {
        std::vector<Legion::FieldID> field_ids_copy{field_ids};
        const Legion::FieldSpace result =
            rt->create_field_space(ctx, field_sizes, field_ids_copy);
        assert(field_ids == field_ids_copy);
        return result;
    }


    template <void (*TASK_PTR)(const Legion::Task *,
                               const std::vector<Legion::PhysicalRegion> &,
                               Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id,
                              const std::string &task_name,
                              bool is_inner, bool is_leaf, bool verbose) {
        if (verbose) {
            std::cout << "[LegionSolvers] Registering task " << task_name
                      << " with ID " << task_id << "." << std::endl;
        }
        Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
        registrar.add_constraint(
            Legion::ProcessorConstraint{Legion::Processor::LOC_PROC}
        );
        registrar.set_inner(is_inner);
        registrar.set_leaf(is_leaf);
        Legion::Runtime::preregister_task_variant<TASK_PTR>(
            registrar, task_name.c_str()
        );
    }


    template <typename RETURN_T,
              RETURN_T (*TASK_PTR)(const Legion::Task *,
                                   const std::vector<Legion::PhysicalRegion> &,
                                   Legion::Context, Legion::Runtime *)>
    void preregister_cpu_task(Legion::TaskID task_id,
                              const std::string &task_name,
                              bool is_inner, bool is_leaf, bool verbose) {
        if (verbose) {
            std::cout << "[LegionSolvers] Registering task " << task_name
                      << " with ID " << task_id << "." << std::endl;
        }
        Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
        registrar.add_constraint(
            Legion::ProcessorConstraint{Legion::Processor::LOC_PROC}
        );
        registrar.set_inner(is_inner);
        registrar.set_leaf(is_leaf);
        if constexpr (std::is_void_v<RETURN_T>) {
            Legion::Runtime::preregister_task_variant<TASK_PTR>(
                registrar, task_name.c_str()
            );
        } else {
            Legion::Runtime::preregister_task_variant<RETURN_T, TASK_PTR>(
                registrar, task_name.c_str()
            );
        }
    }


    template <typename ReturnType,
              template <typename> typename KokkosTaskTemplate>
    void preregister_kokkos_task(Legion::TaskID task_id,
                                 const std::string &task_name,
                                 bool is_inner, bool is_leaf, bool verbose) {

        #ifdef KOKKOS_ENABLE_SERIAL
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos CPU task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                Legion::Processor::LOC_PROC
            });
            registrar.set_inner(is_inner);
            registrar.set_leaf(is_leaf);
            if constexpr (std::is_void_v<ReturnType>) {
                Legion::Runtime::preregister_task_variant<
                    KokkosTaskTemplate<Kokkos::Serial>::task_body
                >(registrar, task_name.c_str());
            } else {
                Legion::Runtime::preregister_task_variant<
                    ReturnType,
                    KokkosTaskTemplate<Kokkos::Serial>::task_body
                >(registrar, task_name.c_str());
            }
        }
        #endif

        #ifdef KOKKOS_ENABLE_OPENMP
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos OMP task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                #ifdef REALM_USE_OPENMP
                    Legion::Processor::OMP_PROC
                #else
                    Legion::Processor::LOC_PROC
                #endif
            });
            registrar.set_inner(is_inner);
            registrar.set_leaf(is_leaf);
            if constexpr (std::is_void_v<ReturnType>) {
                Legion::Runtime::preregister_task_variant<
                    KokkosTaskTemplate<Kokkos::OpenMP>::task_body
                >(registrar, task_name.c_str());
            } else {
                Legion::Runtime::preregister_task_variant<
                    ReturnType,
                    KokkosTaskTemplate<Kokkos::OpenMP>::task_body
                >(registrar, task_name.c_str());
            }
        }
        #endif

        #if defined(KOKKOS_ENABLE_CUDA) and defined(REALM_USE_CUDA)
        {
            if (verbose) {
                std::cout << "[LegionSolvers] Registering Kokkos GPU task "
                          << task_name << " with ID "
                          << task_id << "." << std::endl;
            }
            Legion::TaskVariantRegistrar registrar{task_id, task_name.c_str()};
            registrar.add_constraint(Legion::ProcessorConstraint{
                Legion::Processor::TOC_PROC
            });
            registrar.set_inner(is_inner);
            registrar.set_leaf(is_leaf);
            if constexpr (std::is_void_v<ReturnType>) {
                Legion::Runtime::preregister_task_variant<
                    KokkosTaskTemplate<Kokkos::Cuda>::task_body
                >(registrar, task_name.c_str());
            } else {
                Legion::Runtime::preregister_task_variant<
                    ReturnType,
                    KokkosTaskTemplate<Kokkos::Cuda>::task_body
                >(registrar, task_name.c_str());
            }
        }
        #endif

    }


    struct ProjectionOneLevel final : public Legion::ProjectionFunctor {

        Legion::coord_t index;

        explicit ProjectionOneLevel(Legion::coord_t index) noexcept :
            index(index) {}

        virtual bool is_functional() const noexcept { return true; }

        virtual unsigned get_depth() const noexcept { return 0; }

        using Legion::ProjectionFunctor::project;

        virtual Legion::LogicalRegion project(
            Legion::LogicalPartition upper_bound,
            const Legion::DomainPoint &point,
            const Legion::Domain &launch_domain
        ) override {
            return runtime->get_logical_subregion_by_color(
                upper_bound, point[index]
            );
        }

    }; // struct ProjectionOneLevel


    struct ProjectionTwoLevel final : public Legion::ProjectionFunctor {

        Legion::coord_t i;
        Legion::coord_t j;

        explicit ProjectionTwoLevel(Legion::coord_t i, Legion::coord_t j)
            noexcept : i(i), j(j) {}

        virtual bool is_functional() const noexcept { return true; }

        virtual unsigned get_depth() const noexcept { return 1; }

        using Legion::ProjectionFunctor::project;

        virtual Legion::LogicalRegion project(
            Legion::LogicalPartition upper_bound,
            const Legion::DomainPoint &point,
            const Legion::Domain &launch_domain
        ) override {
            const auto column = runtime->get_logical_subregion_by_color(
                upper_bound, point[i]
            );
            const auto partition = runtime->get_logical_partition_by_color(
                column, LEGION_SOLVERS_DEFAULT_TILE_PARTITION_COLOR
            );
            return runtime->get_logical_subregion_by_color(partition, point[j]);
        }

    }; // struct ProjectionTwoLevel


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

    }; // class RectIterator


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

    }; // class Rects


    template <typename FIELD_TYPE, int DIM, typename COORD_T>
    using AffineReader = Legion::FieldAccessor<
        LEGION_READ_ONLY, FIELD_TYPE, DIM, COORD_T,
        Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
        LEGION_SOLVERS_CHECK_BOUNDS
    >;


    template <typename FIELD_TYPE, int DIM, typename COORD_T>
    using AffineWriter = Legion::FieldAccessor<
        LEGION_WRITE_ONLY, FIELD_TYPE, DIM, COORD_T,
        Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
        LEGION_SOLVERS_CHECK_BOUNDS
    >;


    template <typename FIELD_TYPE, int DIM, typename COORD_T>
    using AffineSumAccessor = Legion::ReductionAccessor<
        Legion::SumReduction<FIELD_TYPE>, false, // non-exclusive
        DIM, COORD_T, Realm::AffineAccessor<FIELD_TYPE, DIM, COORD_T>,
        LEGION_SOLVERS_CHECK_BOUNDS
    >;


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LEGION_UTILITIES_HPP

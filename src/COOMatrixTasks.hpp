#ifndef LEGION_SOLVERS_COO_MATRIX_TASKS_HPP
#define LEGION_SOLVERS_COO_MATRIX_TASKS_HPP

#include <string>

#include <legion.h>

#include "KokkosUtilities.hpp"
#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename ExecutionSpace, typename ENTRY_T,
              int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct KokkosCOOMatvecFunctor {

        Realm::AffineAccessor<
            Legion::Point<RANGE_DIM>, KERNEL_DIM, Legion::coord_t
        > i_reader;

        Realm::AffineAccessor<
            Legion::Point<DOMAIN_DIM>, KERNEL_DIM, Legion::coord_t
        > j_reader;

        Realm::AffineAccessor<ENTRY_T, KERNEL_DIM, Legion::coord_t>
        entry_reader;

        Realm::AffineAccessor<ENTRY_T, DOMAIN_DIM, Legion::coord_t>
        input_reader;

        Realm::AffineAccessor<ENTRY_T, RANGE_DIM, Legion::coord_t>
        output_writer;

        KOKKOS_INLINE_FUNCTION void operator()(int a) const {
            const Legion::Point<1> index{a};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b) const {
            const Legion::Point<2> index{a, b};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, int c) const {
            const Legion::Point<3> index{a, b, c};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d
        ) const {
            const Legion::Point<4> index{a, b, c, d};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e
        ) const {
            const Legion::Point<5> index{a, b, c, d, e};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f
        ) const {
            const Legion::Point<6> index{a, b, c, d, e, f};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f, int g
        ) const {
            const Legion::Point<7> index{a, b, c, d, e, f, g};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d,
            int e, int f, int g, int h
        ) const {
            const Legion::Point<8> index{a, b, c, d, e, f, g, h};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e,
            int f, int g, int h, int k
        ) const {
            const Legion::Point<9> index{a, b, c, d, e, f, g, h, k};
            const Legion::Point<RANGE_DIM> i = i_reader[index];
            const Legion::Point<DOMAIN_DIM> j = j_reader[index];
            const ENTRY_T entry = entry_reader[index];
            output_writer[i] += entry * input_reader[j];
        }

    }; // struct KokkosCOOMatvecFunctor


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct COOMatvecTask : TaskTDDD<COO_MATVEC_TASK_BLOCK_ID,
                                    COOMatvecTask, ENTRY_T,
                                    KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

        static constexpr const char *task_base_name() { return "coo_matvec"; }

        static constexpr bool is_leaf = true;

        using ReturnType = void;

        template <typename ExecutionSpace>
        struct KokkosTaskBody {

            static void body(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context ctx, Legion::Runtime *rt) {

                // COOMatvecTask::announce(typeid(ExecutionSpace), ctx, rt);

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
                    Legion::SumReduction<ENTRY_T>, true,
                    RANGE_DIM, Legion::coord_t,
                    Realm::AffineAccessor<ENTRY_T, RANGE_DIM, Legion::coord_t>
                > output_writer{
                    output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>
                };

                for (Legion::RectInDomainIterator<KERNEL_DIM> it{coo_matrix};
                     it(); ++it) {
                    const Legion::Rect<KERNEL_DIM> rect = *it;
                    Kokkos::parallel_for(
                        KokkosRangeFactory<ExecutionSpace, KERNEL_DIM>::create(
                            rect, ctx, rt
                        ),
                        KokkosCOOMatvecFunctor<
                            ExecutionSpace, ENTRY_T,
                            KERNEL_DIM, DOMAIN_DIM, RANGE_DIM
                        >{
                            i_reader.accessor, j_reader.accessor,
                            entry_reader.accessor,
                            input_reader.accessor, output_writer.accessor
                        }
                    );
                }
            }

        }; // struct KokkosTaskBody

    }; // struct COOMatvecTask


    // template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    // struct COORmatvecTask : TaskTDDD<COO_RMATVEC_TASK_BLOCK_ID,
    //                                  COORmatvecTask, ENTRY_T,
    //                                  KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

    //     static std::string task_name() { return "coo_rmatvec"; }

    //     static void task(const Legion::Task *task,
    //                      const std::vector<Legion::PhysicalRegion> &regions,
    //                      Legion::Context ctx, Legion::Runtime *rt) {

    //         assert(regions.size() == 3);
    //         const auto &output_vec = regions[0];
    //         const auto &coo_matrix = regions[1];
    //         const auto &input_vec = regions[2];

    //         assert(task->regions.size() == 3);
    //         const auto &output_req = task->regions[0];
    //         const auto &matrix_req = task->regions[1];
    //         const auto &input_req = task->regions[2];

    //         assert(output_req.privilege_fields.size() == 1);
    //         const Legion::FieldID output_fid =
    //             *output_req.privilege_fields.begin();

    //         assert(input_req.privilege_fields.size() == 1);
    //         const Legion::FieldID input_fid =
    //             *input_req.privilege_fields.begin();

    //         assert(matrix_req.privilege_fields.size() == 3);
    //         assert(task->arglen == 3 * sizeof(Legion::FieldID));
    //         const Legion::FieldID *argptr =
    //             reinterpret_cast<const Legion::FieldID *>(task->args);
    //         const Legion::FieldID fid_i = argptr[0];
    //         const Legion::FieldID fid_j = argptr[1];
    //         const Legion::FieldID fid_entry = argptr[2];

    //         const Legion::FieldAccessor<
    //             LEGION_READ_ONLY, Legion::Point<RANGE_DIM>, KERNEL_DIM
    //         > i_reader{coo_matrix, fid_i};

    //         const Legion::FieldAccessor<
    //             LEGION_READ_ONLY, Legion::Point<DOMAIN_DIM>, KERNEL_DIM
    //         > j_reader{coo_matrix, fid_j};

    //         const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, KERNEL_DIM>
    //         entry_reader{coo_matrix, fid_entry};

    //         const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, RANGE_DIM>
    //         input_reader{input_vec, input_fid};

    //         const Legion::ReductionAccessor<
    //             Legion::SumReduction<ENTRY_T>, false,
    //             DOMAIN_DIM, Legion::coord_t,
    //             Realm::AffineAccessor<ENTRY_T, DOMAIN_DIM, Legion::coord_t>
    //         > output_writer{output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>};

    //         for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix};
    //              iter(); ++iter) {
    //             const Legion::Point<RANGE_DIM> i{i_reader[*iter]};
    //             const Legion::Point<DOMAIN_DIM> j{j_reader[*iter]};
    //             const ENTRY_T entry = entry_reader[*iter];
    //             output_writer[j] <<= entry * input_reader[i];
    //         }
    //     }

    // }; // struct COORmatvecTask


    // template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    // struct COOPrintTask : TaskTDDD<COO_PRINT_TASK_BLOCK_ID,
    //                                COOPrintTask, ENTRY_T,
    //                                KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

    //     static std::string task_name() { return "coo_print"; }

    //     static void task(const Legion::Task *task,
    //                      const std::vector<Legion::PhysicalRegion> &regions,
    //                      Legion::Context ctx, Legion::Runtime *rt) {

    //         assert(regions.size() == 1);
    //         const auto &coo_matrix = regions[0];

    //         assert(task->arglen == 3 * sizeof(Legion::FieldID));
    //         const Legion::FieldID *argptr =
    //             reinterpret_cast<const Legion::FieldID *>(task->args);
    //         const Legion::FieldID fid_i = argptr[0];
    //         const Legion::FieldID fid_j = argptr[1];
    //         const Legion::FieldID fid_entry = argptr[2];

    //         const Legion::FieldAccessor<
    //             LEGION_READ_ONLY, Legion::Point<RANGE_DIM>, KERNEL_DIM
    //         > i_reader{coo_matrix, fid_i};

    //         const Legion::FieldAccessor<
    //             LEGION_READ_ONLY, Legion::Point<DOMAIN_DIM>, KERNEL_DIM
    //         > j_reader{coo_matrix, fid_j};

    //         const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, KERNEL_DIM>
    //         entry_reader{coo_matrix, fid_entry};

    //         std::cout << "[LegionSolvers] Printing COO matrix:" << std::endl;
    //         for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix};
    //              iter(); ++iter) {
    //             const Legion::Point<RANGE_DIM> i{i_reader[*iter]};
    //             const Legion::Point<DOMAIN_DIM> j{j_reader[*iter]};
    //             const ENTRY_T entry = entry_reader[*iter];
    //             std::cout << "[LegionSolvers]     ";
    //             if (task->is_index_space) {
    //                 std::cout << task->index_point << ' ';
    //             }
    //             std::cout << *iter << ": (" << i << ", " << j << "), "
    //                       << entry << std::endl;
    //         }
    //     }

    // }; // struct COOPrintTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_COO_MATRIX_TASKS_HPP

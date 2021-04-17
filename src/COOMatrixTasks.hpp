#ifndef LEGION_SOLVERS_COO_MATRIX_TASKS_HPP
#define LEGION_SOLVERS_COO_MATRIX_TASKS_HPP

#include <string>

#include <legion.h>

#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct COOMatvecTask : TaskTDDD<COO_MATVEC_TASK_BLOCK_ID, ENTRY_T,
                                    KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

        static std::string task_name() { return "coo_matvec"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx, Legion::Runtime *rt) {

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
                LEGION_READ_ONLY, Legion::Point<RANGE_DIM>, KERNEL_DIM
            > i_reader{coo_matrix, fid_i};

            const Legion::FieldAccessor<
                LEGION_READ_ONLY, Legion::Point<DOMAIN_DIM>, KERNEL_DIM
            > j_reader{coo_matrix, fid_j};

            const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, KERNEL_DIM>
            entry_reader{coo_matrix, fid_entry};

            const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, DOMAIN_DIM>
            input_reader{input_vec, input_fid};

            const Legion::ReductionAccessor<
                Legion::SumReduction<ENTRY_T>, false,
                RANGE_DIM, Legion::coord_t,
                Realm::AffineAccessor<ENTRY_T, RANGE_DIM, Legion::coord_t>
            > output_writer{output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>};

            for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix};
                 iter(); ++iter) {
                const Legion::Point<RANGE_DIM> i{i_reader[*iter]};
                const Legion::Point<DOMAIN_DIM> j{j_reader[*iter]};
                const ENTRY_T entry = entry_reader[*iter];
                output_writer[i] <<= entry * input_reader[j];
            }
        }

    }; // struct COOMatvecTask


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct COORmatvecTask : TaskTDDD<COO_RMATVEC_TASK_BLOCK_ID, ENTRY_T,
                                     KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

        static std::string task_name() { return "coo_rmatvec"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx, Legion::Runtime *rt) {

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
                LEGION_READ_ONLY, Legion::Point<RANGE_DIM>, KERNEL_DIM
            > i_reader{coo_matrix, fid_i};

            const Legion::FieldAccessor<
                LEGION_READ_ONLY, Legion::Point<DOMAIN_DIM>, KERNEL_DIM
            > j_reader{coo_matrix, fid_j};

            const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, KERNEL_DIM>
            entry_reader{coo_matrix, fid_entry};

            const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, RANGE_DIM>
            input_reader{input_vec, input_fid};

            const Legion::ReductionAccessor<
                Legion::SumReduction<ENTRY_T>, false,
                DOMAIN_DIM, Legion::coord_t,
                Realm::AffineAccessor<ENTRY_T, DOMAIN_DIM, Legion::coord_t>
            > output_writer{output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>};

            for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix};
                 iter(); ++iter) {
                const Legion::Point<RANGE_DIM> i{i_reader[*iter]};
                const Legion::Point<DOMAIN_DIM> j{j_reader[*iter]};
                const ENTRY_T entry = entry_reader[*iter];
                output_writer[j] <<= entry * input_reader[i];
            }
        }

    }; // struct COORmatvecTask


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct COOPrintTask : TaskTDDD<COO_PRINT_TASK_BLOCK_ID, ENTRY_T,
                                   KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

        static std::string task_name() { return "coo_print"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx, Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto &coo_matrix = regions[0];

            assert(task->arglen == 3 * sizeof(Legion::FieldID));
            const Legion::FieldID *argptr =
                reinterpret_cast<const Legion::FieldID *>(task->args);
            const Legion::FieldID fid_i = argptr[0];
            const Legion::FieldID fid_j = argptr[1];
            const Legion::FieldID fid_entry = argptr[2];

            const Legion::FieldAccessor<
                LEGION_READ_ONLY, Legion::Point<RANGE_DIM>, KERNEL_DIM
            > i_reader{coo_matrix, fid_i};

            const Legion::FieldAccessor<
                LEGION_READ_ONLY, Legion::Point<DOMAIN_DIM>, KERNEL_DIM
            > j_reader{coo_matrix, fid_j};

            const Legion::FieldAccessor<LEGION_READ_ONLY, ENTRY_T, KERNEL_DIM>
            entry_reader{coo_matrix, fid_entry};

            std::cout << "[LegionSolvers] Printing COO matrix:" << std::endl;
            for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix};
                 iter(); ++iter) {
                const Legion::Point<RANGE_DIM> i{i_reader[*iter]};
                const Legion::Point<DOMAIN_DIM> j{j_reader[*iter]};
                const ENTRY_T entry = entry_reader[*iter];
                std::cout << "[LegionSolvers]     ";
                if (task->is_index_space) {
                    std::cout << task->index_point << ' ';
                }
                std::cout << *iter << ": (" << i << ", " << j << "), "
                          << entry << std::endl;
            }
        }

    }; // struct COOPrintTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_COO_MATRIX_TASKS_HPP

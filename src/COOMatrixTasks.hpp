#ifndef LEGION_SOLVERS_COO_MATRIX_TASKS_HPP
#define LEGION_SOLVERS_COO_MATRIX_TASKS_HPP

#include <string>

#include <legion.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct COOMatvecTask : TaskTDDD<COO_MATVEC_TASK_BLOCK_ID, T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

        static std::string task_name() { return "coo_matvec"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 3);
            const auto &output_vec = regions[0];
            const auto &coo_matrix = regions[1];
            const auto &input_vec = regions[2];

            assert(task->regions.size() == 3);
            const auto &output_req = task->regions[0];
            const auto &matrix_req = task->regions[1];
            const auto &input_req = task->regions[2];

            assert(output_req.privilege_fields.size() == 1);
            const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

            assert(matrix_req.privilege_fields.size() == 3);

            assert(input_req.privilege_fields.size() == 1);
            const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

            assert(task->arglen == 3 * sizeof(Legion::FieldID));
            const Legion::FieldID *argptr = reinterpret_cast<const Legion::FieldID *>(task->args);

            const Legion::FieldID fid_i = argptr[0];
            const Legion::FieldID fid_j = argptr[1];
            const Legion::FieldID fid_entry = argptr[2];

            const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::Point<RANGE_DIM>, KERNEL_DIM> i_reader{coo_matrix,
                                                                                                         fid_i};
            const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::Point<DOMAIN_DIM>, KERNEL_DIM> j_reader{coo_matrix,
                                                                                                          fid_j};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, KERNEL_DIM> entry_reader{coo_matrix, fid_entry};

            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DOMAIN_DIM> input_reader{input_vec, input_fid};
            const Legion::FieldAccessor<LEGION_READ_WRITE, T, RANGE_DIM> output_writer{output_vec, output_fid};

            for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix}; iter(); ++iter) {
                const Legion::Point<RANGE_DIM> i{i_reader[*iter]};
                const Legion::Point<DOMAIN_DIM> j{j_reader[*iter]};
                const T entry = entry_reader[*iter];
                output_writer[i] = output_writer[i] + entry * input_reader[j];
            }
        }

    }; // struct COOMatvecTask


    template <typename T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct COORmatvecTask : TaskTDDD<COO_RMATVEC_TASK_BLOCK_ID, T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

        static std::string task_name() { return "coo_rmatvec"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 3);
            const auto &output_vec = regions[0];
            const auto &coo_matrix = regions[1];
            const auto &input_vec = regions[2];

            assert(task->regions.size() == 3);
            const auto &output_req = task->regions[0];
            const auto &matrix_req = task->regions[1];
            const auto &input_req = task->regions[2];

            assert(output_req.privilege_fields.size() == 1);
            const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

            assert(matrix_req.privilege_fields.size() == 3);

            assert(input_req.privilege_fields.size() == 1);
            const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

            assert(task->arglen == 3 * sizeof(Legion::FieldID));
            const Legion::FieldID *argptr = reinterpret_cast<const Legion::FieldID *>(task->args);

            const Legion::FieldID fid_i = argptr[0];
            const Legion::FieldID fid_j = argptr[1];
            const Legion::FieldID fid_entry = argptr[2];

            const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::Point<RANGE_DIM>, KERNEL_DIM> i_reader{coo_matrix,
                                                                                                         fid_i};
            const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::Point<DOMAIN_DIM>, KERNEL_DIM> j_reader{coo_matrix,
                                                                                                          fid_j};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, KERNEL_DIM> entry_reader{coo_matrix, fid_entry};

            const Legion::FieldAccessor<LEGION_READ_ONLY, T, RANGE_DIM> input_reader{input_vec, input_fid};
            const Legion::FieldAccessor<LEGION_READ_WRITE, T, DOMAIN_DIM> output_writer{output_vec, output_fid};

            for (Legion::PointInDomainIterator<KERNEL_DIM> iter{coo_matrix}; iter(); ++iter) {
                const Legion::Point<RANGE_DIM> i{i_reader[*iter]};
                const Legion::Point<DOMAIN_DIM> j{j_reader[*iter]};
                const T entry = entry_reader[*iter];
                output_writer[j] = output_writer[j] + entry * input_reader[i];
            }
        }

    }; // struct COORmatvecTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_COO_MATRIX_TASKS_HPP

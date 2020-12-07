#ifndef LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
#define LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP

#include <string>

#include <legion.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T, int DIM>
    struct AxpyTask : TaskTD<AXPY_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "axpy"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 2);
            const auto &y = regions[0];
            const auto &x = regions[1];

            assert(task->regions.size() == 2);
            const auto &y_req = task->regions[0];
            const auto &x_req = task->regions[1];

            assert(y_req.privilege_fields.size() == 1);
            const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

            assert(x_req.privilege_fields.size() == 1);
            const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

            assert(task->futures.size() == 1);
            const T alpha = task->futures[0].get_result<T>();

            const Legion::FieldAccessor<LEGION_READ_WRITE, T, DIM> y_writer{y, y_fid};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> x_reader{x, x_fid};

            for (Legion::PointInDomainIterator<DIM> iter{y}; iter(); ++iter) {
                y_writer[*iter] = alpha * x_reader[*iter] + y_writer[*iter];
            }
        }

    }; // struct AxpyTask


    template <typename T, int DIM>
    struct XpayTask : TaskTD<XPAY_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "xpay"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 2);
            const auto &y = regions[0];
            const auto &x = regions[1];

            assert(task->regions.size() == 2);
            const auto &y_req = task->regions[0];
            const auto &x_req = task->regions[1];

            assert(y_req.privilege_fields.size() == 1);
            const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

            assert(x_req.privilege_fields.size() == 1);
            const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

            assert(task->futures.size() == 1);
            const T alpha = task->futures[0].get_result<T>();

            const Legion::FieldAccessor<LEGION_READ_WRITE, T, DIM> y_writer{y, y_fid};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> x_reader{x, x_fid};
            for (Legion::PointInDomainIterator<DIM> iter{y}; iter(); ++iter) {
                y_writer[*iter] = x_reader[*iter] + alpha * y_writer[*iter];
            }
        }

    }; // struct XpayTask


    template <typename T, int DIM>
    struct DotProductTask : TaskTD<DOT_PRODUCT_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "dot_product"; }

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context ctx,
                      Legion::Runtime *rt) {

            assert(regions.size() == 2);
            const auto &v = regions[0];
            const auto &w = regions[1];

            assert(task->regions.size() == 2);
            const auto &v_req = task->regions[0];
            const auto &w_req = task->regions[1];

            assert(v_req.privilege_fields.size() == 1);
            const Legion::FieldID v_fid = *v_req.privilege_fields.begin();

            assert(w_req.privilege_fields.size() == 1);
            const Legion::FieldID w_fid = *w_req.privilege_fields.begin();

            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> v_reader{v, v_fid};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> w_reader{w, w_fid};

            T result = static_cast<T>(0);
            for (Legion::PointInDomainIterator<DIM> iter{v}; iter(); ++iter) {
                result += v_reader[*iter] * w_reader[*iter];
            }
            return result;
        }

    }; // struct DotProductTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP

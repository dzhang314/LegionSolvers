#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP
#define LEGION_SOLVERS_UTILITY_TASKS_HPP

#include <string>

#include <legion.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T>
    struct AdditionTask : TaskT<ADDITION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() + b.get_result<T>();
        }

    }; // struct AdditionTask


    template <typename T>
    Legion::Future add(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{AdditionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct SubtractionTask : TaskT<SUBTRACTION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() - b.get_result<T>();
        }

    }; // struct SubtractionTask


    template <typename T>
    Legion::Future subtract(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{SubtractionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct NegationTask : TaskT<NEGATION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 1);
            Legion::Future a = task->futures[0];

            return -a.get_result<T>();
        }

    }; // struct NegationTask


    template <typename T>
    Legion::Future negate(Legion::Future a, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{NegationTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct MultiplicationTask : TaskT<MULTIPLICATION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() * b.get_result<T>();
        }

    }; // struct MultiplicationTask


    template <typename T>
    Legion::Future multiply(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{MultiplicationTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct DivisionTask : TaskT<DIVISION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() / b.get_result<T>();
        }

    }; // struct DivisionTask


    template <typename T>
    Legion::Future divide(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{DivisionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <int DIM>
    struct IsNonemptyTask : TaskD<IS_NONEMPTY_TASK_BLOCK_ID, DIM> {

        static bool task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto region = regions[0];

            bool result = false;
            for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) {
                result = true;
                break;
            }
            return result;
        }

    }; // struct IsNonemptyTask


    template <int DIM>
    Legion::Future
    is_nonempty(Legion::LogicalRegion region, Legion::FieldID fid, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{IsNonemptyTask<DIM>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{region, LEGION_READ_ONLY, LEGION_EXCLUSIVE, region});
        launcher.add_field(0, fid);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T, int DIM>
    struct ConstantFillTask : TaskTD<CONSTANT_FILL_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "constant_fill"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto &region = regions[0];

            assert(task->regions.size() == 1);
            const auto &region_req = task->regions[0];

            assert(region_req.privilege_fields.size() == 1);
            const Legion::FieldID fid = *region_req.privilege_fields.begin();

            assert(task->arglen == sizeof(T));
            const T fill_value = *reinterpret_cast<const T *>(task->args);

            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM> entry_writer{region, fid};
            for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) { entry_writer[*iter] = fill_value; }
        }

    }; // struct ConstantFillTask


    template <typename T, int DIM>
    struct CopyTask : TaskTD<COPY_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "copy"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 2);
            const auto &dst = regions[0];
            const auto &src = regions[1];

            assert(task->regions.size() == 2);
            const auto &dst_req = task->regions[0];
            const auto &src_req = task->regions[1];

            assert(dst_req.privilege_fields.size() == 1);
            const Legion::FieldID dst_fid = *dst_req.privilege_fields.begin();

            assert(src_req.privilege_fields.size() == 1);
            const Legion::FieldID src_fid = *src_req.privilege_fields.begin();

            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM> dst_writer{dst, dst_fid};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> src_reader{src, src_fid};

            for (Legion::PointInDomainIterator<DIM> iter{dst}; iter(); ++iter) {
                dst_writer[*iter] = src_reader[*iter];
            }
        }

    }; // struct CopyTask


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


    template <typename T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct CooMatvecTask : TaskTDDD<COO_MATVEC_TASK_BLOCK_ID, T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {

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
    };


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP

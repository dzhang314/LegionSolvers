#ifndef LEGION_SOLVERS_TASKS_HPP
#define LEGION_SOLVERS_TASKS_HPP

#include <legion.h>


enum SolverTaskIDs : Legion::TaskID {
    ZERO_FILL_TASK_ID = 4001,
    COO_MATVEC_TASK_ID = 4002,
    IS_NONEMPTY_TASK_ID = 4003,
    DIVISION_TASK_ID = 4004,
    NEGATION_TASK_ID = 4005,
    AXPY_TASK_ID = 4006,
    XPAY_TASK_ID = 4007,
    DOT_PRODUCT_TASK_ID = 4008,
    COPY_TASK_ID = 4009,
    ADDITION_TASK_ID = 4010,
};


template <typename T, int DIM>
void zero_fill_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx, Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &region = regions[0];

    assert(task->regions.size() == 1);
    const auto &region_req = task->regions[0];

    assert(region_req.privilege_fields.size() == 1);
    const Legion::FieldID fid = *region_req.privilege_fields.begin();

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM> entry_writer{
        region, fid};
    for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) {
        entry_writer[*iter] = static_cast<T>(0);
    }
}


template <typename T, int MATRIX_DIM, int INPUT_DIM, int OUTPUT_DIM>
void coo_matvec_task(const Legion::Task *task,
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
    const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 3);

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

    assert(task->arglen == 3 * sizeof(Legion::FieldID));
    const Legion::FieldID *argptr =
        reinterpret_cast<const Legion::FieldID *>(task->args);

    const Legion::FieldID fid_i = argptr[0];
    const Legion::FieldID fid_j = argptr[1];
    const Legion::FieldID fid_entry = argptr[2];

    const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::Point<OUTPUT_DIM>,
                                MATRIX_DIM>
        i_reader{coo_matrix, fid_i};
    const Legion::FieldAccessor<LEGION_READ_ONLY, Legion::Point<INPUT_DIM>,
                                MATRIX_DIM>
        j_reader{coo_matrix, fid_j};
    const Legion::FieldAccessor<LEGION_READ_ONLY, T, MATRIX_DIM> entry_reader{
        coo_matrix, fid_entry};

    const Legion::FieldAccessor<LEGION_READ_ONLY, T, INPUT_DIM> input_reader{
        input_vec, input_fid};
    const Legion::FieldAccessor<LEGION_READ_WRITE, T, OUTPUT_DIM> output_writer{
        output_vec, output_fid};

    for (Legion::PointInDomainIterator<MATRIX_DIM> iter{coo_matrix}; iter();
         ++iter) {
        const Legion::Point<OUTPUT_DIM> i{i_reader[*iter]};
        const Legion::Point<INPUT_DIM> j{j_reader[*iter]};
        const T entry = entry_reader[*iter];
        output_writer[i] = output_writer[i] + entry * input_reader[j];
    }
}


template <int DIM>
bool is_nonempty_task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context ctx, Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto region = regions[0];

    bool result = false;
    for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) {
        result = true;
        break;
    }
    return result;
}


template <typename T>
T division_task(const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context, Legion::Runtime *) {

    assert(task->futures.size() == 2);
    Legion::Future numerator = task->futures[0];
    Legion::Future denominator = task->futures[1];

    return numerator.get_result<T>() / denominator.get_result<T>();
}


template <typename T>
T negation_task(const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context, Legion::Runtime *) {

    assert(task->futures.size() == 1);
    Legion::Future x = task->futures[0];

    return -x.get_result<T>();
}


template <typename T, int DIM>
void axpy_task(const Legion::Task *task,
               const std::vector<Legion::PhysicalRegion> &regions,
               Legion::Context ctx, Legion::Runtime *runtime) {

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


template <typename T, int DIM>
void xpay_task(const Legion::Task *task,
               const std::vector<Legion::PhysicalRegion> &regions,
               Legion::Context ctx, Legion::Runtime *runtime) {

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


template <typename T, int DIM>
T dot_product_task(const Legion::Task *task,
                   const std::vector<Legion::PhysicalRegion> &regions,
                   Legion::Context ctx, Legion::Runtime *runtime) {

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


template <typename T, int DIM>
void copy_task(const Legion::Task *task,
               const std::vector<Legion::PhysicalRegion> &regions,
               Legion::Context ctx, Legion::Runtime *runtime) {

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

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM> dst_writer{
        dst, dst_fid};
    const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> src_reader{src,
                                                                     src_fid};

    for (Legion::PointInDomainIterator<DIM> iter{dst}; iter(); ++iter) {
        dst_writer[*iter] = +src_reader[*iter]; // TODO: Why is + needed?
    }
}


template <typename T>
T addition_task(const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context, Legion::Runtime *) {

    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];

    return a.get_result<T>() + b.get_result<T>();
}


template <void (*TaskPtr)(const Legion::Task *,
                          const std::vector<Legion::PhysicalRegion> &,
                          Legion::Context, Legion::Runtime *)>
void preregister_cpu_task(Legion::TaskID task_id, const char *task_name) {
    Legion::TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(
        Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
    Legion::Runtime::preregister_task_variant<TaskPtr>(registrar, task_name);
}


template <typename ReturnType,
          ReturnType (*TaskPtr)(const Legion::Task *,
                                const std::vector<Legion::PhysicalRegion> &,
                                Legion::Context, Legion::Runtime *)>
void preregister_cpu_task(Legion::TaskID task_id, const char *task_name) {
    Legion::TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(
        Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
    Legion::Runtime::preregister_task_variant<ReturnType, TaskPtr>(registrar,
                                                                   task_name);
}


template <typename T, int DIM>
void preregister_solver_tasks() {
    preregister_cpu_task<zero_fill_task<T, DIM>>(ZERO_FILL_TASK_ID,
                                                 "zero_fill");
    preregister_cpu_task<coo_matvec_task<T, 1, DIM, DIM>>(COO_MATVEC_TASK_ID,
                                                          "coo_matvec");
    preregister_cpu_task<bool, is_nonempty_task<1>>(IS_NONEMPTY_TASK_ID,
                                                    "is_nonempty");
    preregister_cpu_task<T, division_task<T>>(DIVISION_TASK_ID, "division");
    preregister_cpu_task<T, negation_task<T>>(NEGATION_TASK_ID, "negation");
    preregister_cpu_task<axpy_task<T, DIM>>(AXPY_TASK_ID, "axpy");
    preregister_cpu_task<xpay_task<T, DIM>>(XPAY_TASK_ID, "xpay");
    preregister_cpu_task<T, dot_product_task<T, DIM>>(DOT_PRODUCT_TASK_ID,
                                                      "dot_product");
    preregister_cpu_task<copy_task<T, DIM>>(COPY_TASK_ID, "copy");
    preregister_cpu_task<T, addition_task<T>>(ADDITION_TASK_ID, "addition");
}


#endif // LEGION_SOLVERS_TASKS_HPP

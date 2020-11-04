#ifndef LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP
#define LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP

#include <legion.h>


enum ConjugateGradientSolverFieldIDs : Legion::FieldID {
    FID_CG_P = 300, // step direction
    FID_CG_Q = 301, // conjugate step direction
    FID_CG_R = 302, // residual vector
    FID_CG_X = 303, // solution vector
};

class ConjugateGradientSolver {

  public:
    Legion::LogicalRegion workspace;

    explicit ConjugateGradientSolver(Legion::LogicalRegion rhs,
                                     Legion::FieldID fid_rhs,
                                     Legion::Context ctx, Legion::Runtime *rt) {
        const Legion::IndexSpace workspace_is = rhs.get_index_space();
        const Legion::FieldSpace workspace_fs = rt->create_field_space(ctx);
        Legion::FieldAllocator allocator =
            rt->create_field_allocator(ctx, workspace_fs);
        allocator.allocate_field(sizeof(double), FID_CG_P);
        allocator.allocate_field(sizeof(double), FID_CG_Q);
        allocator.allocate_field(sizeof(double), FID_CG_R);
        allocator.allocate_field(sizeof(double), FID_CG_X);
        workspace = rt->create_logical_region(ctx, workspace_is, workspace_fs);
    }

    Legion::Future dot_product(Legion::FieldID fid_v, Legion::FieldID fid_w,
                               Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{DOT_PRODUCT_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            workspace, READ_ONLY, EXCLUSIVE, workspace});
        launcher.add_field(0, fid_v);
        launcher.add_region_requirement(Legion::RegionRequirement{
            workspace, READ_ONLY, EXCLUSIVE, workspace});
        launcher.add_field(1, fid_w);
        return rt->execute_task(ctx, launcher);
    }

    Legion::Future divide(Legion::Future numer, Legion::Future denom,
                          Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{DIVISION_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(numer);
        launcher.add_future(denom);
        return rt->execute_task(ctx, launcher);
    }

    Legion::Future negate(Legion::Future x, Legion::Context ctx,
                          Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{NEGATION_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(x);
        return rt->execute_task(ctx, launcher);
    }

    void axpy(Legion::FieldID fid_y, Legion::Future alpha,
              Legion::FieldID fid_x, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{AXPY_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            workspace, READ_WRITE, EXCLUSIVE, workspace});
        launcher.add_field(0, fid_y);
        launcher.add_region_requirement(Legion::RegionRequirement{
            workspace, READ_ONLY, EXCLUSIVE, workspace});
        launcher.add_field(1, fid_x);
        launcher.add_future(alpha);
        rt->execute_task(ctx, launcher);
    }

    void xpay(Legion::FieldID fid_y, Legion::Future alpha,
              Legion::FieldID fid_x, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{XPAY_TASK_ID,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            workspace, READ_WRITE, EXCLUSIVE, workspace});
        launcher.add_field(0, fid_y);
        launcher.add_region_requirement(Legion::RegionRequirement{
            workspace, READ_ONLY, EXCLUSIVE, workspace});
        launcher.add_field(1, fid_x);
        launcher.add_future(alpha);
        rt->execute_task(ctx, launcher);
    }

    void solve() {
        // Legion::Future r_norm2 =
        //     planner.dot_product(FID_CG_R, FID_CG_R, workspace);
        // for (int i = 0; i < 16; ++i) {
        //     planner.matmul(FID_CG_Q, FID_CG_P, workspace);
        //     Legion::Future p_norm =
        //         planner.dot_product(FID_CG_P, FID_CG_Q, workspace);
        //     Legion::Future alpha = planner.divide(r_norm2, p_norm);
        //     planner.axpy(sol_fid, FID_CG_P, alpha, stencil_lr, workspace,
        //                  color_space, disjoint_partition,
        //                  workspace_disjoint_partition);
        //     planner.axpy(FID_CG_R, FID_CG_Q, planner.negate(alpha),
        //     workspace,
        //                  color_space, workspace_disjoint_partition);
        //     Legion::Future r_norm2_new =
        //         planner.dot_product(FID_CG_R, FID_CG_R, workspace);
        //     Legion::Future beta = planner.divide(r_norm2_new, r_norm2);
        //     r_norm2 = r_norm2_new;
        //     planner.aypx(FID_CG_P, FID_CG_R, beta, workspace, color_space,
        //                  workspace_disjoint_partition);
        // }
    }

}; // class ConjugateGradientSolver

#endif // LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP

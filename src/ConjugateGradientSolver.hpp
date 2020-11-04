#ifndef LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP
#define LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP

#include <vector>

#include <legion.h>

#include "Planner.hpp"


enum ConjugateGradientSolverFieldIDs : Legion::FieldID {
    FID_CG_P = 300, // step direction
    FID_CG_Q = 301, // conjugate step direction
    FID_CG_R = 302, // residual vector
    FID_CG_X = 303, // solution vector
};


class ConjugateGradientSolver {

  public:
    const Planner &planner;
    std::vector<Legion::LogicalRegion> workspace;
    Legion::Future residual_norm_squared;

    explicit ConjugateGradientSolver(const Planner &planner,
                                     Legion::Context ctx, Legion::Runtime *rt)
        : planner{planner}, workspace{}, residual_norm_squared{} {
        for (const auto &[index_space, index_partition] : planner.dimensions) {
            const Legion::FieldSpace field_space = rt->create_field_space(ctx);
            Legion::FieldAllocator allocator =
                rt->create_field_allocator(ctx, field_space);
            allocator.allocate_field(sizeof(double), FID_CG_P);
            allocator.allocate_field(sizeof(double), FID_CG_Q);
            allocator.allocate_field(sizeof(double), FID_CG_R);
            allocator.allocate_field(sizeof(double), FID_CG_X);
            workspace.push_back(
                rt->create_logical_region(ctx, index_space, field_space));
        }
    }

    void setup(Legion::Context ctx, Legion::Runtime *rt) {
        planner.zero_fill(FID_CG_X, workspace, ctx, rt);
        planner.copy_rhs(FID_CG_P, workspace, ctx, rt);
        planner.copy_rhs(FID_CG_R, workspace, ctx, rt);
        residual_norm_squared =
            planner.dot_product(FID_CG_R, FID_CG_R, workspace, ctx, rt);
    }

    void step(Legion::Context ctx, Legion::Runtime *rt) {
        planner.matvec(FID_CG_Q, FID_CG_P, workspace, ctx, rt);
        Legion::Future p_norm =
            planner.dot_product(FID_CG_P, FID_CG_Q, workspace, ctx, rt);
        Legion::Future alpha =
            planner.divide(residual_norm_squared, p_norm, ctx, rt);
        planner.axpy(FID_CG_X, alpha, FID_CG_P, workspace, ctx, rt);
        planner.axpy(FID_CG_R, planner.negate(alpha, ctx, rt), FID_CG_Q,
                     workspace, ctx, rt);
        Legion::Future r_norm2_new =
            planner.dot_product(FID_CG_R, FID_CG_R, workspace, ctx, rt);
        Legion::Future beta =
            planner.divide(r_norm2_new, residual_norm_squared, ctx, rt);
        residual_norm_squared = r_norm2_new;
        planner.xpay(FID_CG_P, beta, FID_CG_R, workspace, ctx, rt);
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

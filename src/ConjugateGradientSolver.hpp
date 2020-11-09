#ifndef LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP
#define LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP

#include <cmath>
#include <vector>

#include <legion.h>

#include "Planner.hpp"


namespace LegionSolvers {


    class ConjugateGradientSolver {


        const Planner &planner;
        Legion::Future residual_norm_squared;
        int max_iterations = 1'000;
        double residual_threshold = 1.0e-16;


      public:
        std::vector<Legion::LogicalRegion> workspace;
        enum ConjugateGradientSolverFieldIDs : Legion::FieldID {
            FID_CG_P = 300, // step direction
            FID_CG_Q = 301, // conjugate step direction
            FID_CG_R = 302, // residual vector
            FID_CG_X = 303, // solution vector
        };


        explicit ConjugateGradientSolver(const Planner &planner,
                                         Legion::Context ctx,
                                         Legion::Runtime *rt)
            : planner{planner} {
            for (const auto &[index_space, index_partition] :
                 planner.dimensions) {
                const Legion::FieldSpace field_space =
                    rt->create_field_space(ctx);
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


        void set_max_iterations(int n) { max_iterations = n; }


        void set_residual_threshold(double x) { residual_threshold = x; }


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


        void solve(Legion::Context ctx, Legion::Runtime *rt) {
            setup(ctx, rt);
            for (int i = 0; i < max_iterations; ++i) {
                std::cout << "residual: "
                          << std::sqrt(
                                 residual_norm_squared.get_result<double>())
                          << std::endl;
                if (residual_norm_squared.get_result<double>() <=
                    residual_threshold * residual_threshold) {
                    break;
                }
                step(ctx, rt);
            }
        }


    }; // class ConjugateGradientSolver


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_CONJUGATE_GRADIENT_SOLVER_HPP

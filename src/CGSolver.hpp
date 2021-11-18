#ifndef LEGION_SOLVERS_CG_SOLVER_HPP
#define LEGION_SOLVERS_CG_SOLVER_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include <legion.h>

#include "DistributedVector.hpp"
#include "SquarePlanner.hpp"


namespace LegionSolvers {


    template <typename T>
    class CGSolver {

        const SquarePlanner<T> &planner;

    public:

        enum : Legion::FieldID { P, Q, R };

        std::vector<std::vector<std::unique_ptr<DistributedVector<T>>>>
        workspace;

        std::vector<Scalar<T>> residual_norm_squared;

        explicit CGSolver(
            const SquarePlanner<T> &planner
        ) : planner(planner),
            workspace() {
            assert(planner.solution_vectors.size() == planner.rhs_vectors.size());
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            for (const auto &rhs : planner.rhs_vectors) {
                workspace[P].emplace_back(rhs->similar("P"));
                workspace[Q].emplace_back(rhs->similar("Q"));
                workspace[R].emplace_back(rhs->similar("R"));
            }
        }

        void setup() {
            planner.copy_rhs(workspace[P]);
            planner.copy_rhs(workspace[R]);
            residual_norm_squared.push_back(planner.dot(workspace[R], workspace[R]));
        }

        void step() {
            planner.matvec(workspace[Q], workspace[P]);
            Scalar<T> p_norm = planner.dot(workspace[P], workspace[Q]);
            Scalar<T> alpha = residual_norm_squared.back() / p_norm;
            planner.axpy_sol(alpha, workspace[P]);
            planner.axpy(workspace[R], -alpha, workspace[Q]);
            Scalar<T> r_norm2_new = planner.dot(workspace[R], workspace[R]);
            Scalar<T> beta = r_norm2_new / residual_norm_squared.back();
            residual_norm_squared.push_back(r_norm2_new);
            planner.xpay(workspace[P], beta, workspace[R]);
        }

        void solve() {
            setup();
            for (int i = 0; i < 10; ++i) {
                // rt->begin_trace(ctx, 101);
                step();
                // rt->end_trace(ctx, 101);
                // const T r2 = residual_norm_squared[2].get_result<T>();
                // if (print_residual) {
                //     std::cout << "residual: " << std::sqrt(r2) << std::endl;
                // }
                // if (r2 <= residual_threshold * residual_threshold) { break; }
            }
        }

    }; // class CGSolver


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_CG_SOLVER_HPP

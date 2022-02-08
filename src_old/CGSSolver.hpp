#ifndef LEGION_SOLVERS_CGS_SOLVER_HPP
#define LEGION_SOLVERS_CGS_SOLVER_HPP

#include <memory>
#include <vector>

#include <legion.h>

#include "DistributedVector.hpp"
#include "SquarePlanner.hpp"


namespace LegionSolvers {


    template <typename T>
    class CGSSolver {

        const SquarePlanner<T> &planner;

    public:

        enum : Legion::FieldID { P, P_HAT, Q, Q_HAT, R, R_TILDE, U, V_HAT };

        std::vector<std::vector<std::unique_ptr<DistributedVector<T>>>>
        workspace;

        std::vector<Scalar<T>> rho;

        explicit CGSSolver(
            const SquarePlanner<T> &planner
        ) : planner(planner),
            workspace(),
            rho() {
            assert(planner.solution_vectors.size() ==
                   planner.rhs_vectors.size());
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            for (const auto &rhs : planner.rhs_vectors) {
                workspace[P      ].emplace_back(rhs->similar("P"));
                workspace[P_HAT  ].emplace_back(rhs->similar("P_HAT"));
                workspace[Q      ].emplace_back(rhs->similar("Q"));
                workspace[Q_HAT  ].emplace_back(rhs->similar("Q_HAT"));
                workspace[R      ].emplace_back(rhs->similar("R"));
                workspace[R_TILDE].emplace_back(rhs->similar("R_TILDE"));
                workspace[U      ].emplace_back(rhs->similar("U"));
                workspace[V_HAT  ].emplace_back(rhs->similar("V_HAT"));
            }
        }

        void setup() {
            planner.copy_rhs(workspace[R]);
            planner.copy_rhs(workspace[R_TILDE]);
            rho.push_back(planner.dot(workspace[R_TILDE], workspace[R]));
            planner.zero_fill(workspace[P]);
            planner.zero_fill(workspace[Q]);
        }

        void step() {
            // TODO: Something is wrong here, don't know what yet
            // First two iterations come back fine, but third iteration is wrong
            Scalar<T> rho_new = planner.dot(workspace[R_TILDE], workspace[R]);
            Scalar<T> beta = rho_new / rho.back();
            rho.push_back(rho_new);
            planner.copy(workspace[U], workspace[R]);
            planner.axpy(workspace[U], beta, workspace[Q]);
            planner.xpay(workspace[P], beta, workspace[Q]);
            planner.xpay(workspace[P], beta, workspace[U]);
            planner.matvec(workspace[V_HAT], workspace[P]);
            Scalar<T> temp = planner.dot(workspace[R_TILDE], workspace[V_HAT]);
            Scalar<T> alpha = rho.back() / temp;
            planner.copy(workspace[Q], workspace[U]);
            planner.axpy(workspace[Q], -alpha, workspace[V_HAT]);
            planner.copy(workspace[P_HAT], workspace[U]);
            planner.axpy(
                workspace[P_HAT],
                Scalar<T>(static_cast<T>(1), planner.ctx, planner.rt),
                workspace[Q]
            );
            planner.axpy_sol(alpha, workspace[P_HAT]);
            planner.matvec(workspace[Q_HAT], workspace[P_HAT]);
            planner.axpy(workspace[R], -alpha, workspace[Q_HAT]);
        }

    }; // class CGSSolver


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_CGS_SOLVER_HPP

#ifndef LEGION_SOLVERS_BICGSTAB_SOLVER_HPP
#define LEGION_SOLVERS_BICGSTAB_SOLVER_HPP

#include <memory>
#include <vector>

#include <legion.h>

#include "DistributedVector.hpp"
#include "SquarePlanner.hpp"


namespace LegionSolvers {


    template <typename T>
    class BiCGStabSolver {

        const SquarePlanner<T> &planner;

    public:

        enum : Legion::FieldID { P, R, R_TILDE, U, V };

        std::vector<std::vector<std::unique_ptr<DistributedVector<T>>>>
        workspace;

        std::vector<Scalar<T>> rho;
        std::vector<Scalar<T>> alpha;
        std::vector<Scalar<T>> omega;

        explicit BiCGStabSolver(
            const SquarePlanner<T> &planner
        ) : planner(planner),
            workspace(),
            rho(),
            alpha(),
            omega() {
            assert(planner.solution_vectors.size() ==
                   planner.rhs_vectors.size());
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            workspace.emplace_back();
            for (const auto &rhs : planner.rhs_vectors) {
                workspace[P      ].emplace_back(rhs->similar("P"));
                workspace[R      ].emplace_back(rhs->similar("R"));
                workspace[R_TILDE].emplace_back(rhs->similar("R_TILDE"));
                workspace[U      ].emplace_back(rhs->similar("U"));
                workspace[V      ].emplace_back(rhs->similar("V"));
            }
        }

        void setup() {
            planner.copy_rhs(workspace[R]);
            planner.copy_rhs(workspace[R_TILDE]);
            rho.push_back(Scalar<T>(static_cast<T>(1), planner.ctx, planner.rt));
            alpha.push_back(Scalar<T>(static_cast<T>(0), planner.ctx, planner.rt));
            omega.push_back(Scalar<T>(static_cast<T>(1), planner.ctx, planner.rt));
            planner.zero_fill(workspace[P]);
            planner.zero_fill(workspace[V]);
        }

        void step() {
            Scalar<T> rho_new = planner.dot(workspace[R], workspace[R_TILDE]);
            Scalar<T> beta = (rho_new / rho.back()) * (alpha.back() / omega.back());
            rho.push_back(rho_new);
            planner.axpy(workspace[P], -omega.back(), workspace[V]);
            planner.xpay(workspace[P], beta, workspace[R]);
            planner.matvec(workspace[V], workspace[P]);
            Scalar<T> temp = planner.dot(workspace[R_TILDE], workspace[V]);
            alpha.push_back(rho.back() / temp);
            planner.axpy(workspace[R], -alpha.back(), workspace[V]);
            planner.matvec(workspace[U], workspace[R]);
            Scalar<T> r_anorm2 = planner.dot(workspace[R], workspace[U]);
            Scalar<T> u_norm2 = planner.dot(workspace[U], workspace[U]);
            omega.push_back(r_anorm2 / u_norm2);
            planner.axpy_sol(alpha.back(), workspace[P]);
            planner.axpy_sol(omega.back(), workspace[R]);
            planner.axpy(workspace[R], -omega.back(), workspace[U]);
        }

    }; // class BiCGStabSolver


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_BICGSTAB_SOLVER_HPP

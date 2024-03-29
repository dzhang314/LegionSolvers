#ifndef LEGION_SOLVERS_CG_SOLVER_HPP_INCLUDED
#define LEGION_SOLVERS_CG_SOLVER_HPP_INCLUDED

#include <cstddef> // for std::size_t
#include <vector>  // for std::vector

#include "Scalar.hpp"
#include "SquarePlanner.hpp"

namespace LegionSolvers {


template <typename ENTRY_T>
class CGSolver {

    static constexpr std::size_t SOL = 0;
    static constexpr std::size_t RHS = 1;
    static constexpr std::size_t P = 2;
    static constexpr std::size_t Q = 3;
    static constexpr std::size_t R = 4;

public:

    SquarePlanner<ENTRY_T> &planner;
    // TODO: don't store all past residuals -- change this to a circular buffer
    // to store the last N residuals, where N is a constructor parameter
    std::vector<Scalar<ENTRY_T>> residual_norm_squared;
    Scalar<ENTRY_T> negative_one;

public:

    explicit CGSolver(SquarePlanner<ENTRY_T> &planner)
        : planner(planner)
        , residual_norm_squared()
        , negative_one(
              planner.get_context(),
              planner.get_runtime(),
              static_cast<ENTRY_T>(-1)
          ) {
        planner.allocate_workspace(3);
        planner.copy(P, RHS);
        planner.copy(R, RHS);
        residual_norm_squared.push_back(planner.dot(R, R));
    }

    void step() {
        planner.matvec(Q, P);
        Scalar<ENTRY_T> p_norm = planner.dot(P, Q);
        Scalar<ENTRY_T> r_norm2_old = residual_norm_squared.back();
        planner.axpy(SOL, r_norm2_old, p_norm, P);
        planner.axpy(R, negative_one, r_norm2_old, p_norm, Q);
        Scalar<ENTRY_T> r_norm2_new = planner.dot(R, R);
        residual_norm_squared.push_back(r_norm2_new);
        planner.xpay(P, r_norm2_new, r_norm2_old, R);
    }

}; // class CGSolver


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_CG_SOLVER_HPP_INCLUDED

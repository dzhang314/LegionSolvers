#ifndef LEGION_SOLVERS_BICGSTAB_SOLVER_HPP_INCLUDED
#define LEGION_SOLVERS_BICGSTAB_SOLVER_HPP_INCLUDED

#include <cstddef> // for std::size_t
#include <vector>  // for std::vector

#include "Scalar.hpp"
#include "SquarePlanner.hpp"

namespace LegionSolvers {


template <typename ENTRY_T>
class BiCGStabSolver {

    static constexpr std::size_t SOL = 0;
    static constexpr std::size_t RHS = 1;
    static constexpr std::size_t P = 2;
    static constexpr std::size_t R = 3;
    static constexpr std::size_t R_TILDE = 4;
    static constexpr std::size_t U = 5;
    static constexpr std::size_t V = 6;

public:

    SquarePlanner<ENTRY_T> &planner;
    std::vector<Scalar<ENTRY_T>> rho;
    std::vector<Scalar<ENTRY_T>> alpha;
    std::vector<Scalar<ENTRY_T>> omega;
    Scalar<ENTRY_T> negative_one;
    Scalar<ENTRY_T> zero;
    Scalar<ENTRY_T> one;

public:

    explicit BiCGStabSolver(SquarePlanner<ENTRY_T> &planner)
        : planner(planner)
        , rho()
        , alpha()
        , omega()
        , negative_one(
              planner.get_context(),
              planner.get_runtime(),
              static_cast<ENTRY_T>(-1)
          )
        , zero(
              planner.get_context(),
              planner.get_runtime(),
              static_cast<ENTRY_T>(0)
          )
        , one(planner.get_context(),
              planner.get_runtime(),
              static_cast<ENTRY_T>(1)) {
        planner.allocate_workspace(5);
        planner.copy(R, RHS);
        planner.copy(R_TILDE, RHS);
        rho.push_back(one);
        alpha.push_back(zero);
        omega.push_back(one);
        planner.zero_fill(P);
        planner.zero_fill(V);
    }

    void step() {
        Scalar<ENTRY_T> rho_new = planner.dot(R, R_TILDE);
        Scalar<ENTRY_T> beta =
            (rho_new / rho.back()) * (alpha.back() / omega.back());
        rho.push_back(rho_new);
        planner.axpy(P, -omega.back(), V);
        planner.xpay(P, beta, R);
        planner.matvec(V, P);
        Scalar<ENTRY_T> temp = planner.dot(R_TILDE, V);
        planner.axpy(R, negative_one, rho.back(), temp, V);
        alpha.push_back(rho.back() / temp);
        planner.matvec(U, R);
        Scalar<ENTRY_T> r_anorm2 = planner.dot(R, U);
        Scalar<ENTRY_T> u_norm2 = planner.dot(U, U);
        omega.push_back(r_anorm2 / u_norm2);
        planner.axpy(SOL, alpha.back(), P);
        planner.axpy(SOL, omega.back(), R);
        planner.axpy(R, -omega.back(), U);
    }

}; // class BiCGStabSolver


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_BICGSTAB_SOLVER_HPP_INCLUDED

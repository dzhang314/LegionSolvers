#ifndef LEGION_SOLVERS_GMRES_SOLVER_HPP_INCLUDED
#define LEGION_SOLVERS_GMRES_SOLVER_HPP_INCLUDED

#include <cstddef> // for std::size_t
#include <utility> // for std::move
#include <vector>  // for std::vector

#include "Scalar.hpp"
#include "SquarePlanner.hpp"
#include "UtilityTasks.hpp"

namespace LegionSolvers {


template <typename ENTRY_T>
class GMRESSolver {

    static constexpr std::size_t SOL = 0;
    static constexpr std::size_t RHS = 1;

public:

    SquarePlanner<ENTRY_T> &planner;
    std::size_t restart;
    Scalar<ENTRY_T> negative_one;
    std::vector<std::vector<Scalar<ENTRY_T>>> inner_products;
    std::vector<std::vector<Scalar<ENTRY_T>>> triangular_matrix;
    std::vector<Scalar<ENTRY_T>> coefficients;

public:

    explicit GMRESSolver(SquarePlanner<ENTRY_T> &planner, std::size_t restart)
        : planner(planner)
        , restart(restart)
        , negative_one(
              planner.get_context(),
              planner.get_runtime(),
              static_cast<ENTRY_T>(-1)
          )
        , inner_products()
        , triangular_matrix()
        , coefficients() {

        planner.allocate_workspace(restart + 1);

        for (std::size_t i = 0; i <= restart; ++i) {
            std::vector<Scalar<ENTRY_T>> row;
            for (std::size_t j = 0; j < restart; ++j) {
                row.emplace_back(
                    planner.get_context(),
                    planner.get_runtime(),
                    static_cast<ENTRY_T>(0)
                );
            }
            inner_products.push_back(std::move(row));
        }

        for (std::size_t i = 0; i < restart; ++i) {
            std::vector<Scalar<ENTRY_T>> row;
            for (std::size_t j = 0; j < restart; ++j) {
                row.emplace_back(
                    planner.get_context(),
                    planner.get_runtime(),
                    static_cast<ENTRY_T>(0)
                );
            }
            triangular_matrix.push_back(std::move(row));
        }

        for (std::size_t j = 0; j < restart; ++j) {
            coefficients.emplace_back(
                planner.get_context(),
                planner.get_runtime(),
                static_cast<ENTRY_T>(0)
            );
        }
    }

    static constexpr std::size_t krylov_basis(std::size_t i) noexcept {
        return i + 2;
    }

    void step() {

        // compute residual vector
        planner.matvec(krylov_basis(0), SOL);
        planner.xpay(krylov_basis(0), negative_one, RHS);
        const auto residual_norm =
            planner.dot(krylov_basis(0), krylov_basis(0)).rsqrt();
        planner.scal(krylov_basis(0), residual_norm);

        for (std::size_t j = 0; j < restart; ++j) {
            planner.matvec(krylov_basis(j + 1), krylov_basis(j));
            for (std::size_t k = 0; k <= j; ++k) {
                inner_products[k][j] =
                    planner.dot(krylov_basis(k), krylov_basis(j + 1));
                planner.axpy(
                    krylov_basis(j + 1), -inner_products[k][j], krylov_basis(k)
                );
            }
            const auto d =
                planner.dot(krylov_basis(j + 1), krylov_basis(j + 1));
            inner_products[j + 1][j] = d.sqrt();
            if (j + 1 < restart) {
                planner.scal(krylov_basis(j + 1), d.rsqrt());
            }
        }

        Legion::TaskLauncher launcher(
            DummyTask<ENTRY_T>::task_id, Legion::TaskArgument{}
        );
        for (std::size_t i = 0; i <= restart; ++i) {
            for (std::size_t j = 0; j < restart; ++j) {
                launcher.add_future(inner_products[i][j].get_future());
            }
        }
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        Legion::Future future = planner.get_runtime()->execute_task(
            planner.get_context(), launcher
        );
        Scalar<ENTRY_T> coeff{
            planner.get_context(), planner.get_runtime(), future};

        for (std::size_t j = 0; j < restart; ++j) {
            planner.axpy(SOL, coeff, krylov_basis(j));
        }
    }

}; // class GMRESSolver


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_GMRES_SOLVER_HPP_INCLUDED

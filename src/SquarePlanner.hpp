#ifndef LEGION_SOLVERS_SQUARE_PLANNER_HPP
#define LEGION_SOLVERS_SQUARE_PLANNER_HPP

#include <cassert>
#include <cstddef>
#include <tuple>
#include <vector>

#include <legion.h>

#include "DistributedVector.hpp"
#include "MaterializedLinearOperator.hpp"


namespace LegionSolvers {


    template <typename T>
    class SquarePlanner {

    public:

        const Legion::Context ctx;
        Legion::Runtime *const rt;

        std::vector<DistributedVector<T> *> solution_vectors;
        std::vector<DistributedVector<T> *> rhs_vectors;
        std::vector<std::tuple<
            std::size_t, std::size_t, const MaterializedLinearOperator<T> *
        >> operators;
        std::vector<Legion::IndexSpaceT<3>> tile_maps;

        explicit SquarePlanner(Legion::Context ctx, Legion::Runtime *rt) :
            ctx(ctx),
            rt(rt),
            solution_vectors(),
            rhs_vectors(),
            operators(),
            tile_maps() {}

        std::size_t add_solution_vector(DistributedVector<T> &sol) {
            const std::size_t index = solution_vectors.size();
            solution_vectors.push_back(&sol);
            if (rhs_vectors.size() > index) {
                const DistributedVector<T> &rhs = *rhs_vectors[index];
                assert(sol.get_index_space() == rhs.get_index_space());
                assert(sol.get_color_space() == rhs.get_color_space());
                assert(sol.get_index_partition() == rhs.get_index_partition());
            }
            return index;
        }

        std::size_t add_rhs_vector(DistributedVector<T> &rhs) {
            const std::size_t index = rhs_vectors.size();
            rhs_vectors.push_back(&rhs);
            if (solution_vectors.size() > index) {
                const DistributedVector<T> &sol = *solution_vectors[index];
                assert(rhs.get_index_space() == sol.get_index_space());
                assert(rhs.get_color_space() == sol.get_color_space());
                assert(rhs.get_index_partition() == sol.get_index_partition());
            }
            return index;
        }

        void add_operator(
            std::size_t sol_index, std::size_t rhs_index,
            const MaterializedLinearOperator<T> &matrix
        ) {
            assert(solution_vectors.size() > sol_index);
            assert(rhs_vectors.size() > rhs_index);
            operators.emplace_back(sol_index, rhs_index, &matrix);
            tile_maps.emplace_back(matrix.compute_nonempty_tiles(
                solution_vectors[sol_index]->get_index_partition(),
                rhs_vectors[rhs_index]->get_index_partition(),
                ctx, rt
            ));
        }

        void zero_fill(
            const std::vector<std::unique_ptr<DistributedVector<T>>> &dst
        ) const {
            const std::size_t n = dst.size();
            for (std::size_t i = 0; i < n; ++i) {
                dst[i]->zero_fill();
            }
        }

        void copy(
            const std::vector<std::unique_ptr<DistributedVector<T>>> &dst,
            const std::vector<std::unique_ptr<DistributedVector<T>>> &src
        ) const {
            const std::size_t n = dst.size();
            assert(n == src.size());
            for (std::size_t i = 0; i < n; ++i) {
                *dst[i] = *src[i];
            }
        }

        void copy_rhs(
            const std::vector<std::unique_ptr<DistributedVector<T>>> &dst
        ) const {
            const std::size_t n = dst.size();
            assert(n == rhs_vectors.size());
            for (std::size_t i = 0; i < n; ++i) {
                *dst[i] = *rhs_vectors[i];
            }
        }

        Scalar<T> dot(
            const std::vector<std::unique_ptr<DistributedVector<T>>> &v,
            const std::vector<std::unique_ptr<DistributedVector<T>>> &w
        ) const {
            const std::size_t n = v.size();
            assert(n == w.size());
            Scalar<T> result{static_cast<T>(0), ctx, rt};
            for (std::size_t i = 0; i < n; ++i) {
                result = result + v[i]->dot(*w[i]);
            }
            return result;
        }

        void axpy_sol(
            const Scalar<T> &alpha,
            const std::vector<std::unique_ptr<DistributedVector<T>>> &src
        ) const {
            const std::size_t n = solution_vectors.size();
            assert(n == src.size());
            for (std::size_t i = 0; i < n; ++i) {
                solution_vectors[i]->axpy(alpha, *src[i]);
            }
        }

        void axpy(
            const std::vector<std::unique_ptr<DistributedVector<T>>> &dst,
            const Scalar<T> &alpha,
            const std::vector<std::unique_ptr<DistributedVector<T>>> &src
        ) const {
            const std::size_t n = dst.size();
            assert(n == src.size());
            for (std::size_t i = 0; i < n; ++i) {
                dst[i]->axpy(alpha, *src[i]);
            }
        }

        void xpay(
            const std::vector<std::unique_ptr<DistributedVector<T>>> &dst,
            const Scalar<T> &alpha,
            const std::vector<std::unique_ptr<DistributedVector<T>>> &src
        ) const {
            const std::size_t n = dst.size();
            assert(n == src.size());
            for (std::size_t i = 0; i < n; ++i) {
                dst[i]->xpay(alpha, *src[i]);
            }
        }

        void matvec(
            const std::vector<std::unique_ptr<DistributedVector<T>>> &dst,
            const std::vector<std::unique_ptr<DistributedVector<T>>> &src
        ) const {
            for (const auto &dst_piece : dst) {
                *dst_piece = static_cast<T>(0);
            }
            for (std::size_t i = 0; i < operators.size(); ++i) {
                const auto &[dst_index, src_index, op] = operators[i];
                op->matvec(*dst[dst_index], *src[src_index], tile_maps[i]);
            }
        }

    }; // class SquarePlanner


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_SQUARE_PLANNER_HPP

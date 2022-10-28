#ifndef LEGION_SOLVERS_SQUARE_PLANNER_HPP_INCLUDED
#define LEGION_SOLVERS_SQUARE_PLANNER_HPP_INCLUDED

#include <vector> // for std::vector

#include <legion.h> // for Legion::*

namespace LegionSolvers {


template <typename ENTRY_T>
class SquarePlanner {

    const Legion::Context ctx;
    Legion::Runtime *const rt;

    Legion::Context get_context() const { return ctx; }
    Legion::Runtime *get_runtime() const { return rt; }


}; // class SquarePlanner


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_SQUARE_PLANNER_HPP_INCLUDED

#ifndef LEGION_SOLVERS_LINEAR_OPERATOR_HPP
#define LEGION_SOLVERS_LINEAR_OPERATOR_HPP

#include <legion.h>

#include "DistributedVector.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class LinearOperator {

    public:

        virtual void matvec(
            DistributedVector<ENTRY_T> &output_vector,
            const DistributedVector<ENTRY_T> &input_vector
        ) const = 0;

        virtual void rmatvec(
            DistributedVector<ENTRY_T> &output_vector,
            const DistributedVector<ENTRY_T> &input_vector
        ) const = 0;

        virtual void print() const = 0;

        virtual ~LinearOperator() = 0;

    }; // class LinearOperator


    LinearOperator::~LinearOperator() {}


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LINEAR_OPERATOR_HPP

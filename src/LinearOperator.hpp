#ifndef LEGION_SOLVERS_LINEAR_OPERATOR_HPP
#define LEGION_SOLVERS_LINEAR_OPERATOR_HPP

#include <legion.h>


namespace LegionSolvers {


    class LinearOperator {


      public:
        virtual void matvec(Legion::LogicalRegion output_vector,
                            Legion::FieldID output_fid,
                            Legion::LogicalRegion input_vector,
                            Legion::FieldID input_fid,
                            Legion::Context ctx,
                            Legion::Runtime *rt) const = 0;


        virtual ~LinearOperator() = 0;


    }; // class LinearOperator


    LinearOperator::~LinearOperator() {}


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LINEAR_OPERATOR_HPP

#ifndef LEGION_SOLVERS_ABSTRACT_VECTOR_HPP
#define LEGION_SOLVERS_ABSTRACT_VECTOR_HPP

#include <legion.h>


namespace LegionSolvers {


    template <typename ENTRY_T>
    class AbstractVector {


    public:

        Legion::IndexSpace get_index_space() const  = 0;
        Legion::FieldID get_field_id() const = 0;
        Legion::LogicalRegion get_logical_region() const = 0;


    }; // class AbstractVector


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_ABSTRACT_VECTOR_HPP

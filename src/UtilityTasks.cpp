#include "UtilityTasks.hpp"

#include <iostream> // for std::cout, std::endl

#include "LibraryOptions.hpp" // for LEGION_SOLVERS_USE_*

using LegionSolvers::AddScalarTask;
using LegionSolvers::DivideScalarTask;
using LegionSolvers::MultiplyScalarTask;
using LegionSolvers::NegateScalarTask;
using LegionSolvers::PrintIndexTask;
using LegionSolvers::PrintScalarTask;
using LegionSolvers::SubtractScalarTask;


template <typename T>
int PrintScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    assert((task->futures.size() == 1) || (task->futures.size() == 2));
    Legion::Future x = task->futures[0];
    std::cout << x.get_result<T>() << std::endl;
    return 0;
}


template <typename T>
T NegateScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    assert(task->futures.size() == 1);
    Legion::Future x = task->futures[0];
    return -x.get_result<T>();
}


template <typename T>
T AddScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() + y.get_result<T>();
}


template <typename T>
T SubtractScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() - y.get_result<T>();
}


template <typename T>
T MultiplyScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() * y.get_result<T>();
}


template <typename T>
T DivideScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() / y.get_result<T>();
}


template <int DIM>
void PrintIndexTask<DIM>::task_body(LEGION_SOLVERS_TASK_ARGS) {

    const Legion::DomainPoint index_point = task->index_point;

    assert(regions.size() == 1);
    const auto &dummy = regions[0];

    assert(task->regions.size() == 1);
    const auto &dummy_req = task->regions[0];

    if (task->arglen == 0) {
        for (Legion::PointInDomainIterator<DIM> iter(dummy); iter(); ++iter) {
            std::cout << index_point << ' ' << *iter << '\n';
        }
    } else {
        const std::string name(reinterpret_cast<const char *>(task->args));
        for (Legion::PointInDomainIterator<DIM> iter(dummy); iter(); ++iter) {
            std::cout << name << ' ' << index_point << ' ' << *iter << '\n';
        }
    }
    std::cout << std::flush;
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    template int PrintScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float NegateScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float AddScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float SubtractScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float MultiplyScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float DivideScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    template int PrintScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double NegateScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double AddScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double SubtractScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double MultiplyScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double DivideScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
#endif // LEGION_SOLVERS_USE_F64
#if LEGION_SOLVERS_MAX_DIM >= 1
    template void PrintIndexTask<1>::task_body(LEGION_SOLVERS_TASK_ARGS);
#endif // LEGION_SOLVERS_MAX_DIM >= 1
#if LEGION_SOLVERS_MAX_DIM >= 2
    template void PrintIndexTask<2>::task_body(LEGION_SOLVERS_TASK_ARGS);
#endif // LEGION_SOLVERS_MAX_DIM >= 2
#if LEGION_SOLVERS_MAX_DIM >= 3
    template void PrintIndexTask<3>::task_body(LEGION_SOLVERS_TASK_ARGS);
#endif // LEGION_SOLVERS_MAX_DIM >= 3
// clang-format on

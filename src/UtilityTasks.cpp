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
int PrintScalarTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert((task->futures.size() == 1) || (task->futures.size() == 2));
    Legion::Future x = task->futures[0];
    std::cout << x.get_result<T>() << std::endl;
    return 0;
}


template <typename T>
T NegateScalarTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(task->futures.size() == 1);
    Legion::Future x = task->futures[0];
    return -x.get_result<T>();
}


template <typename T>
T AddScalarTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() + y.get_result<T>();
}


template <typename T>
T SubtractScalarTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() - y.get_result<T>();
}


template <typename T>
T MultiplyScalarTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() * y.get_result<T>();
}


template <typename T>
T DivideScalarTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future x = task->futures[0];
    Legion::Future y = task->futures[1];
    return x.get_result<T>() / y.get_result<T>();
}


template <int DIM>
void LegionSolvers::PrintIndexTask<DIM>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    const Legion::DomainPoint index_point = task->index_point;

    assert(regions.size() == 1);
    const auto &dummy = regions[0];

    assert(task->regions.size() == 1);
    const auto &dummy_req = task->regions[0];

    if (task->arglen == 0) {
        for (Legion::PointInDomainIterator<DIM> iter{dummy}; iter(); ++iter) {
            std::cout << index_point << ' ' << *iter << '\n';
        }
    } else {
        const std::string name{reinterpret_cast<const char *>(task->args)};
        for (Legion::PointInDomainIterator<DIM> iter{dummy}; iter(); ++iter) {
            std::cout << name << ' ' << index_point << ' ' << *iter << '\n';
        }
    }
    std::cout << std::flush;
}


#ifdef LEGION_SOLVERS_USE_FLOAT
template int PrintScalarTask<float>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template float NegateScalarTask<float>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template float AddScalarTask<float>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template float SubtractScalarTask<float>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template float MultiplyScalarTask<float>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template float DivideScalarTask<float>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
#endif // LEGION_SOLVERS_USE_FLOAT


#ifdef LEGION_SOLVERS_USE_DOUBLE
template int PrintScalarTask<double>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template double NegateScalarTask<double>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template double AddScalarTask<double>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template double SubtractScalarTask<double>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template double MultiplyScalarTask<double>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
template double DivideScalarTask<double>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
);
#endif // LEGION_SOLVERS_USE_DOUBLE


// TODO: guards
template void PrintIndexTask<1>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void PrintIndexTask<2>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void PrintIndexTask<3>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

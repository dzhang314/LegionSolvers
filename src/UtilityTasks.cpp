#include "UtilityTasks.hpp"

#include <iostream>
#include <random>
#include <tuple>


template <typename T>
T LegionSolvers::AdditionTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() + b.get_result<T>();
}


template <typename T>
T LegionSolvers::SubtractionTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() - b.get_result<T>();
}


template <typename T>
T LegionSolvers::NegationTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 1);
    Legion::Future a = task->futures[0];
    return -a.get_result<T>();
}


template <typename T>
T LegionSolvers::MultiplicationTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() * b.get_result<T>();
}


template <typename T>
T LegionSolvers::DivisionTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() / b.get_result<T>();
}


template <typename T, int DIM>
void LegionSolvers::RandomFillTask<T, DIM>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // RandomFillTask::announce_cpu(task->index_point, ctx, rt);

    assert(regions.size() == 1);
    const auto &region = regions[0];

    assert(task->regions.size() == 1);
    const auto &region_req = task->regions[0];

    assert(region_req.privilege_fields.size() == 1);
    const Legion::FieldID fid = *region_req.privilege_fields.begin();

    assert(task->arglen == 2 * sizeof(T));
    const T *arg_ptr = reinterpret_cast<const T *>(task->args);
    const T low = arg_ptr[0];
    const T high = arg_ptr[1];

    std::random_device rng{};
    std::uniform_real_distribution<T> entry_dist{low, high};

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM>
    entry_writer{region, fid};

    for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) {
        entry_writer[*iter] = entry_dist(rng);
    }
}


template <typename T, int DIM>
void LegionSolvers::PrintVectorTask<T, DIM>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // PrintVectorTask::announce_cpu(task->index_point, ctx, rt);

    assert(regions.size() == 1);
    const auto &vector = regions[0];

    assert(task->regions.size() == 1);
    const auto &vector_req = task->regions[0];

    assert(vector_req.privilege_fields.size() == 1);
    const Legion::FieldID vector_fid =
        *vector_req.privilege_fields.begin();

    const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM>
    entry_reader{vector, vector_fid};

    if (task->arglen == 0) {
        for (Legion::PointInDomainIterator<DIM> iter{vector}; iter(); ++iter) {
            std::cout << task->index_point << ' '
                      << *iter << ": " << entry_reader[*iter] << '\n';
        }
    } else {
        const std::string name{reinterpret_cast<const char *>(task->args)};
        for (Legion::PointInDomainIterator<DIM> iter{vector}; iter(); ++iter) {
            std::cout << name << ' ' << task->index_point << ' '
                      << *iter << ": " << entry_reader[*iter] << '\n';
        }
    }
    std::cout << std::flush;
}


[[maybe_unused]]
static constexpr auto instantiations = std::make_tuple(

    LegionSolvers::AdditionTask<__half>::task_body,
    LegionSolvers::AdditionTask<float>::task_body,
    LegionSolvers::AdditionTask<double>::task_body,
    LegionSolvers::AdditionTask<long double>::task_body,
    LegionSolvers::AdditionTask<__float128>::task_body,

    LegionSolvers::SubtractionTask<__half>::task_body,
    LegionSolvers::SubtractionTask<float>::task_body,
    LegionSolvers::SubtractionTask<double>::task_body,
    LegionSolvers::SubtractionTask<long double>::task_body,
    LegionSolvers::SubtractionTask<__float128>::task_body,

    LegionSolvers::NegationTask<__half>::task_body,
    LegionSolvers::NegationTask<float>::task_body,
    LegionSolvers::NegationTask<double>::task_body,
    LegionSolvers::NegationTask<long double>::task_body,
    LegionSolvers::NegationTask<__float128>::task_body,

    LegionSolvers::MultiplicationTask<__half>::task_body,
    LegionSolvers::MultiplicationTask<float>::task_body,
    LegionSolvers::MultiplicationTask<double>::task_body,
    LegionSolvers::MultiplicationTask<long double>::task_body,
    LegionSolvers::MultiplicationTask<__float128>::task_body,

    LegionSolvers::DivisionTask<__half>::task_body,
    LegionSolvers::DivisionTask<float>::task_body,
    LegionSolvers::DivisionTask<double>::task_body,
    LegionSolvers::DivisionTask<long double>::task_body,
    LegionSolvers::DivisionTask<__float128>::task_body,

    LegionSolvers::RandomFillTask<float, 1>::task_body,
    LegionSolvers::RandomFillTask<float, 2>::task_body,
    LegionSolvers::RandomFillTask<float, 3>::task_body,
    LegionSolvers::RandomFillTask<double, 1>::task_body,
    LegionSolvers::RandomFillTask<double, 2>::task_body,
    LegionSolvers::RandomFillTask<double, 3>::task_body,
    LegionSolvers::RandomFillTask<long double, 1>::task_body,
    LegionSolvers::RandomFillTask<long double, 2>::task_body,
    LegionSolvers::RandomFillTask<long double, 3>::task_body,

    LegionSolvers::PrintVectorTask<float, 1>::task_body,
    LegionSolvers::PrintVectorTask<float, 2>::task_body,
    LegionSolvers::PrintVectorTask<float, 3>::task_body,
    LegionSolvers::PrintVectorTask<double, 1>::task_body,
    LegionSolvers::PrintVectorTask<double, 2>::task_body,
    LegionSolvers::PrintVectorTask<double, 3>::task_body,
    LegionSolvers::PrintVectorTask<long double, 1>::task_body,
    LegionSolvers::PrintVectorTask<long double, 2>::task_body,
    LegionSolvers::PrintVectorTask<long double, 3>::task_body

);

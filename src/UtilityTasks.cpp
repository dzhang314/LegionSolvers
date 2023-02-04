#include "UtilityTasks.hpp"

#include <array>    // for std::array
#include <cmath>    // for std::sqrt
#include <cstddef>  // for std::size_t
#include <iostream> // for std::cout, std::endl
#include <random>   // for std::random_device, std::mt19937, std::seed_seq

#include "LegionUtilities.hpp" // for AffineWriter
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*

using LegionSolvers::AddScalarTask;
using LegionSolvers::DivideScalarTask;
using LegionSolvers::DummyTask;
using LegionSolvers::MultiplyScalarTask;
using LegionSolvers::NegateScalarTask;
using LegionSolvers::PrintIndexTask;
using LegionSolvers::PrintScalarTask;
using LegionSolvers::RandomFillTask;
using LegionSolvers::RSqrtScalarTask;
using LegionSolvers::SqrtScalarTask;
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


template <typename T>
T SqrtScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    using std::sqrt;
    assert(task->futures.size() == 1);
    Legion::Future x = task->futures[0];
    return sqrt(x.get_result<T>());
}


template <typename T>
T RSqrtScalarTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    using std::sqrt;
    assert(task->futures.size() == 1);
    Legion::Future x = task->futures[0];
    return static_cast<T>(1) / sqrt(x.get_result<T>());
}


template <typename T>
T DummyTask<T>::task_body(LEGION_SOLVERS_TASK_ARGS) {
    return static_cast<T>(1);
}


template <int DIM, typename COORD_T>
void PrintIndexTask<DIM, COORD_T>::task_body(LEGION_SOLVERS_TASK_ARGS) {

    const Legion::DomainPoint index_point = task->index_point;

    assert(regions.size() == 1);
    const auto &dummy = regions[0];

    assert(task->regions.size() == 1);

    if (task->arglen == 0) {
        for (Legion::PointInDomainIterator<DIM, COORD_T> iter(dummy); iter();
             ++iter) {
            std::cout << index_point << ' ' << *iter << '\n';
        }
    } else {
        const std::string name(reinterpret_cast<const char *>(task->args));
        for (Legion::PointInDomainIterator<DIM, COORD_T> iter(dummy); iter();
             ++iter) {
            std::cout << name << ' ' << index_point << ' ' << *iter << '\n';
        }
    }
    std::cout << std::flush;
}


inline std::mt19937 seeded_mersenne_twister() {
    constexpr std::size_t seed_size =
        std::mt19937::state_size * sizeof(typename std::mt19937::result_type);
    std::random_device entropy_source;
    constexpr std::size_t seed_len =
        (seed_size - 1) / sizeof(entropy_source()) + 1;
    std::array<std::random_device::result_type, seed_len> seed_data;
    for (std::size_t i = 0; i < seed_len; ++i) {
        seed_data[i] = entropy_source();
    }
    std::seed_seq seed(seed_data.begin(), seed_data.end());
    return std::mt19937(seed);
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void RandomFillTask<ENTRY_T, DIM, COORD_T>::task_body(LEGION_SOLVERS_TASK_ARGS
) {
    assert(regions.size() == 1);
    const auto &region = regions[0];

    assert(task->regions.size() == 1);
    const auto &req = task->regions[0];

    assert(req.privilege_fields.size() == 1);
    const Legion::FieldID fid = *(req.privilege_fields.begin());

    AffineWriter<ENTRY_T, DIM, COORD_T> writer(region, fid);

    std::mt19937 rng = seeded_mersenne_twister();
    std::uniform_real_distribution<ENTRY_T> dist(
        static_cast<ENTRY_T>(0), static_cast<ENTRY_T>(1)
    );

    for (Legion::PointInDomainIterator<DIM, COORD_T> iter{region}; iter();
         ++iter) {
        writer[*iter] = dist(rng);
    }
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    template int PrintScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float NegateScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float AddScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float SubtractScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float MultiplyScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float DivideScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float SqrtScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float RSqrtScalarTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template float DummyTask<float>::task_body(LEGION_SOLVERS_TASK_ARGS);
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    template int PrintScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double NegateScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double AddScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double SubtractScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double MultiplyScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double DivideScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double SqrtScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double RSqrtScalarTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
    template double DummyTask<double>::task_body(LEGION_SOLVERS_TASK_ARGS);
#endif // LEGION_SOLVERS_USE_F64
#ifdef LEGION_SOLVERS_USE_S32_INDICES
    #if LEGION_SOLVERS_MAX_DIM >= 1
        template void PrintIndexTask<1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 1
    #if LEGION_SOLVERS_MAX_DIM >= 2
        template void PrintIndexTask<2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 2
    #if LEGION_SOLVERS_MAX_DIM >= 3
        template void PrintIndexTask<3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 3
#endif // LEGION_SOLVERS_USE_S32_INDICES
#ifdef LEGION_SOLVERS_USE_U32_INDICES
    #if LEGION_SOLVERS_MAX_DIM >= 1
        template void PrintIndexTask<1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 1
    #if LEGION_SOLVERS_MAX_DIM >= 2
        template void PrintIndexTask<2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 2
    #if LEGION_SOLVERS_MAX_DIM >= 3
        template void PrintIndexTask<3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 3
#endif // LEGION_SOLVERS_USE_U32_INDICES
#ifdef LEGION_SOLVERS_USE_S64_INDICES
    #if LEGION_SOLVERS_MAX_DIM >= 1
        template void PrintIndexTask<1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 1
    #if LEGION_SOLVERS_MAX_DIM >= 2
        template void PrintIndexTask<2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 2
    #if LEGION_SOLVERS_MAX_DIM >= 3
        template void PrintIndexTask<3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
    #endif // LEGION_SOLVERS_MAX_DIM >= 3
#endif // LEGION_SOLVERS_USE_S64_INDICES
#ifdef LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void RandomFillTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void RandomFillTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void RandomFillTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void RandomFillTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void RandomFillTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void RandomFillTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void RandomFillTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void RandomFillTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void RandomFillTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void RandomFillTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void RandomFillTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void RandomFillTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void RandomFillTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void RandomFillTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void RandomFillTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void RandomFillTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void RandomFillTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void RandomFillTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F64
// clang-format on

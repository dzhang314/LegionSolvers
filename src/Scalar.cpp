#include "Scalar.hpp"

#include "LibraryOptions.hpp" // for LEGION_SOLVERS_MAPPER_ID
#include "UtilityTasks.hpp"   // for *ScalarTask

using LegionSolvers::Scalar;


template <typename T>
Scalar<T> Scalar<T>::operator+() const {
    return *this;
}


template <typename T>
Scalar<T> Scalar<T>::operator-() const {
    assert(!futuremap.exists());
    Legion::TaskLauncher launcher(
        NegateScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(future);
    return Scalar<T>{ctx, rt, rt->execute_task(ctx, launcher)};
}


template <typename T>
Scalar<T> Scalar<T>::operator+(const Scalar<T> &rhs) const {
    assert(!futuremap.exists());
    Legion::TaskLauncher launcher(
        AddScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(future);
    launcher.add_future(rhs.future);
    return Scalar<T>{ctx, rt, rt->execute_task(ctx, launcher)};
}


template <typename T>
Scalar<T> Scalar<T>::operator-(const Scalar<T> &rhs) const {
    assert(!futuremap.exists());
    Legion::TaskLauncher launcher(
        SubtractScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(future);
    launcher.add_future(rhs.future);
    return Scalar<T>{ctx, rt, rt->execute_task(ctx, launcher)};
}


template <typename T>
Scalar<T> Scalar<T>::operator*(const Scalar<T> &rhs) const {
    assert(!futuremap.exists());
    Legion::TaskLauncher launcher(
        MultiplyScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(future);
    launcher.add_future(rhs.future);
    return Scalar<T>{ctx, rt, rt->execute_task(ctx, launcher)};
}


template <typename T>
Scalar<T> Scalar<T>::operator/(const Scalar<T> &rhs) const {
    assert(!futuremap.exists());
    Legion::TaskLauncher launcher(
        DivideScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(future);
    launcher.add_future(rhs.future);
    return Scalar<T>{ctx, rt, rt->execute_task(ctx, launcher)};
}


template <typename T>
Scalar<T> Scalar<T>::sqrt() const {
    assert(!futuremap.exists());
    Legion::TaskLauncher launcher(
        SqrtScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(future);
    return Scalar<T>{ctx, rt, rt->execute_task(ctx, launcher)};
}


template <typename T>
Scalar<T> Scalar<T>::rsqrt() const {
    assert(!futuremap.exists());
    Legion::TaskLauncher launcher(
        RSqrtScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(future);
    return Scalar<T>{ctx, rt, rt->execute_task(ctx, launcher)};
}


template <typename T>
Legion::Future Scalar<T>::print() const {
    Legion::TaskLauncher launcher(
        PrintScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(get_future());
    return rt->execute_task(ctx, launcher);
}


template <typename T>
Legion::Future Scalar<T>::print(Legion::Future dummy) const {
    Legion::TaskLauncher launcher(
        PrintScalarTask<T>::task_id, Legion::TaskArgument()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(get_future());
    launcher.add_future(dummy);
    return rt->execute_task(ctx, launcher);
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    template Scalar<float> Scalar<float>::operator+() const;
    template Scalar<float> Scalar<float>::operator-() const;
    template Scalar<float> Scalar<float>::operator+(const Scalar<float> &) const;
    template Scalar<float> Scalar<float>::operator-(const Scalar<float> &) const;
    template Scalar<float> Scalar<float>::operator*(const Scalar<float> &) const;
    template Scalar<float> Scalar<float>::operator/(const Scalar<float> &) const;
    template Scalar<float> Scalar<float>::sqrt() const;
    template Scalar<float> Scalar<float>::rsqrt() const;
    template Legion::Future Scalar<float>::print() const;
    template Legion::Future Scalar<float>::print(Legion::Future) const;
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    template Scalar<double> Scalar<double>::operator+() const;
    template Scalar<double> Scalar<double>::operator-() const;
    template Scalar<double> Scalar<double>::operator+(const Scalar<double> &) const;
    template Scalar<double> Scalar<double>::operator-(const Scalar<double> &) const;
    template Scalar<double> Scalar<double>::operator*(const Scalar<double> &) const;
    template Scalar<double> Scalar<double>::operator/(const Scalar<double> &) const;
    template Scalar<double> Scalar<double>::sqrt() const;
    template Scalar<double> Scalar<double>::rsqrt() const;
    template Legion::Future Scalar<double>::print() const;
    template Legion::Future Scalar<double>::print(Legion::Future) const;
#endif // LEGION_SOLVERS_USE_F64
// clang-format on

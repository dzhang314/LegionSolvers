#ifndef LEGION_SOLVERS_SCALAR_HPP_INCLUDED
#define LEGION_SOLVERS_SCALAR_HPP_INCLUDED

#include <legion.h> // for Legion::*

namespace LegionSolvers {


template <typename T>
class Scalar {

    const Legion::Context ctx;
    Legion::Runtime *const rt;
    Legion::Future future;

public:

    explicit Scalar(
        Legion::Context ctx, Legion::Runtime *rt, const Legion::Future &future
    )
        : ctx(ctx), rt(rt), future(future) {}

    explicit Scalar(Legion::Context ctx, Legion::Runtime *rt, const T &value)
        : ctx(ctx), rt(rt), future(Legion::Future::from_value(rt, value)) {}

    Scalar(const Scalar &) = default; // NOTE: should not be explicit

    Scalar &operator=(const Scalar &rhs) {
        future = rhs.future;
        return *this; // no need to overwrite ctx or rt
    }

    Legion::Future get_future() const { return future; }

    T get_value() const { return future.get_result<T>(); }

    Scalar operator+() const;

    Scalar operator-() const;

    Scalar operator+(const Scalar &rhs) const;

    Scalar operator-(const Scalar &rhs) const;

    Scalar operator*(const Scalar &rhs) const;

    Scalar operator/(const Scalar &rhs) const;

    Legion::Future print() const;

    Legion::Future print(Legion::Future dummy) const;

}; // class Scalar


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_SCALAR_HPP_INCLUDED

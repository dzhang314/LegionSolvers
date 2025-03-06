#ifndef LEGION_SOLVERS_SCALAR_HPP_INCLUDED
#define LEGION_SOLVERS_SCALAR_HPP_INCLUDED

#include <utility> // for std::move

#include <legion.h> // for Legion::*

namespace LegionSolvers {

struct ScalarPackInfo {
  int32_t idx = 0;
  bool fm = false;
  int32_t preceding_futures = 0;
  int32_t preceding_futuremaps = 0;
};

template <typename T>
class Scalar {

    const Legion::Context ctx;
    Legion::Runtime *const rt;
    Legion::Future future;
    Legion::FutureMap futuremap;

public:

    Scalar() = delete;

    explicit Scalar(
        Legion::Context ctx, Legion::Runtime *rt, const Legion::Future &future
    )
        : ctx(ctx)
        , rt(rt)
        , future(future) {}

    explicit Scalar(
        Legion::Context ctx, Legion::Runtime *rt, const Legion::FutureMap &futuremap
    )
        : ctx(ctx)
        , rt(rt)
        , futuremap(futuremap) {}

    explicit Scalar(Legion::Context ctx, Legion::Runtime *rt, const T &value)
        : ctx(ctx)
        , rt(rt)
        , future(Legion::Future::from_value(rt, value)) {}

    Scalar(const Scalar &) = default;

    Scalar(Scalar &&) = default;

    Scalar &operator=(const Scalar &rhs) {
        future = rhs.future;
        return *this; // no need to overwrite ctx or rt
    }

    Scalar &operator=(Scalar &&rhs) {
        future = std::move(rhs.future);
        return *this; // no need to overwrite ctx or rt
    }

    Legion::Future get_future() const { 
      if (futuremap.exists()) {
        return futuremap[0];
      }
      return future;
    }

    T get_value() const { 
      Legion::Future fut = get_future();
      return fut.get_result<T>();
    }

    Scalar operator+() const;

    Scalar operator-() const;

    Scalar operator+(const Scalar &rhs) const;

    Scalar operator-(const Scalar &rhs) const;

    Scalar operator*(const Scalar &rhs) const;

    Scalar operator/(const Scalar &rhs) const;

    Scalar sqrt() const;

    Scalar rsqrt() const;

    Legion::Future print() const;

    Legion::Future print(Legion::Future dummy) const;

    void add_to_launcher(Legion::IndexTaskLauncher& launcher, ScalarPackInfo& info, int32_t idx) const {
      info.idx = idx;
      info.fm = futuremap.exists();
      info.preceding_futures = launcher.futures.size();
      info.preceding_futuremaps = launcher.point_futures.size();
      if (futuremap.exists()) {
        launcher.point_futures.push_back(futuremap);
      } else {
        launcher.add_future(future);
      }
    }

    static void construct_argument_permutation(const Legion::IndexTaskLauncher& launcher, const ScalarPackInfo* infos, int32_t* perm, int N) {
      for (size_t i = 0; i < N; i++) {
        const auto& info = infos[i];
	if (info.fm) {
          perm[info.idx] = launcher.futures.size() + info.preceding_futuremaps;
	} else {
          perm[info.idx] = info.preceding_futures;
	}
      }
    }
}; // class Scalar


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_SCALAR_HPP_INCLUDED

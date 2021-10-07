#include <typeinfo>



    template <typename ExecutionSpace, typename T, int N>
    struct KokkosXpayFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, T, N> y_view;
        const T alpha;
        const KokkosConstOffsetView<ExecutionSpace, T, N> x_view;

        explicit KokkosXpayFunctor(
            Realm::AffineAccessor<T, N, Legion::coord_t> y_accessor,
            const T &a,
            Realm::AffineAccessor<T, N, Legion::coord_t> x_accessor
        ) : y_view{y_accessor}, alpha{a}, x_view{x_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(int a) const {
            y_view(a) = x_view(a) + alpha * y_view(a);
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b) const {
            y_view(a, b) = x_view(a, b) + alpha * y_view(a, b);
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, int c) const {
            y_view(a, b, c) = x_view(a, b, c) + alpha * y_view(a, b, c);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d
        ) const {
            y_view(a, b, c, d) =
                x_view(a, b, c, d) +
                alpha * y_view(a, b, c, d);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e
        ) const {
            y_view(a, b, c, d, e) =
                x_view(a, b, c, d, e) +
                alpha * y_view(a, b, c, d, e);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f
        ) const {
            y_view(a, b, c, d, e, f) =
                x_view(a, b, c, d, e, f) +
                alpha * y_view(a, b, c, d, e, f);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f, int g
        ) const {
            y_view(a, b, c, d, e, f, g) =
                x_view(a, b, c, d, e, f, g) +
                alpha * y_view(a, b, c, d, e, f, g);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d,
            int e, int f, int g, int h
        ) const {
            y_view(a, b, c, d, e, f, g, h) =
                x_view(a, b, c, d, e, f, g, h) +
                alpha * y_view(a, b, c, d, e, f, g, h);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e,
            int f, int g, int h, int i
        ) const {
            y_view(a, b, c, d, e, f, g, h, i) =
                x_view(a, b, c, d, e, f, g, h, i) +
                alpha * y_view(a, b, c, d, e, f, g, h, i);
        }

    }; // struct KokkosXpayFunctor


    template <typename T, int N>
    struct XpayTask : TaskTD<XPAY_TASK_BLOCK_ID, XpayTask, T, N> {

        static constexpr const char *task_base_name() { return "xpay"; }

        static constexpr bool is_leaf = true;

        using ReturnType = void;

        template <typename ExecutionSpace>
        struct KokkosTaskBody {

            static void body(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context ctx, Legion::Runtime *rt) {

                // XpayTask::announce(typeid(ExecutionSpace), ctx, rt);

                assert(regions.size() == 2);
                const auto &y = regions[0];
                const auto &x = regions[1];

                assert(task->regions.size() == 2);
                const auto &y_req = task->regions[0];
                const auto &x_req = task->regions[1];

                assert(y_req.privilege_fields.size() == 1);
                const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

                assert(x_req.privilege_fields.size() == 1);
                const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

                assert(task->futures.size() == 1);
                const T alpha = task->futures[0].get_result<T>();

                Legion::FieldAccessor<
                    LEGION_READ_WRITE, T, N, Legion::coord_t,
                    Realm::AffineAccessor<T, N, Legion::coord_t>
                > y_writer{y, y_fid};

                Legion::FieldAccessor<
                    LEGION_READ_ONLY, T, N, Legion::coord_t,
                    Realm::AffineAccessor<T, N, Legion::coord_t>
                > x_reader{x, x_fid};

                const Legion::Domain y_domain = rt->get_index_space_domain(
                    ctx, y_req.region.get_index_space()
                );

                const Legion::Domain x_domain = rt->get_index_space_domain(
                    ctx, x_req.region.get_index_space()
                );

                assert(y_domain == x_domain);

                for (Legion::RectInDomainIterator<N> it{y_domain}; it(); ++it) {
                    const Legion::Rect<N> rect = *it;
                    Kokkos::parallel_for(
                        KokkosRangeFactory<ExecutionSpace, N>::create(
                            rect, ctx, rt
                        ),
                        KokkosXpayFunctor<ExecutionSpace, T, N>{
                            y_writer.accessor, alpha, x_reader.accessor
                        }
                    );
                }
            }

        }; // struct KokkosTaskBody

    }; // struct XpayTask


    template <typename ExecutionSpace, typename T, int N>
    struct KokkosDotProductFunctor {

        using value_type = T;

        const KokkosConstOffsetView<ExecutionSpace, T, N> v_view;
        const KokkosConstOffsetView<ExecutionSpace, T, N> w_view;

        explicit KokkosDotProductFunctor(
            Realm::AffineAccessor<T, N, Legion::coord_t> v_accessor,
            Realm::AffineAccessor<T, N, Legion::coord_t> w_accessor
        ) : v_view{v_accessor}, w_view{w_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(int a, T &acc) const {
            acc += v_view(a) * w_view(a);
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, T &acc) const {
            acc += v_view(a, b) * w_view(a, b);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, T &acc
        ) const {
            acc += v_view(a, b, c) * w_view(a, b, c);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, T &acc
        ) const {
            acc += v_view(a, b, c, d) * w_view(a, b, c, d);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, T &acc
        ) const {
            acc += v_view(a, b, c, d, e) * w_view(a, b, c, d, e);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f) * w_view(a, b, c, d, e, f);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f, int g, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f, g) * w_view(a, b, c, d, e, f, g);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d,
            int e, int f, int g, int h, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f, g, h) *
                   w_view(a, b, c, d, e, f, g, h);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e,
            int f, int g, int h, int i, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f, g, h, i) *
                   w_view(a, b, c, d, e, f, g, h, i);
        }

    }; // struct KokkosDotProductFunctor


    template <typename T, int N>
    struct DotProductTask : TaskTD<DOT_PRODUCT_TASK_BLOCK_ID,
                                   DotProductTask, T, N> {

        static constexpr const char *task_base_name() { return "dot_product"; }

        static constexpr bool is_leaf = true;

        using ReturnType = T;

        template <typename ExecutionSpace>
        struct KokkosTaskBody {

            static T body(const Legion::Task *task,
                          const std::vector<Legion::PhysicalRegion> &regions,
                          Legion::Context ctx, Legion::Runtime *rt) {

                // DotProductTask::announce(typeid(ExecutionSpace), ctx, rt);

                assert(regions.size() == 2);
                const auto &v = regions[0];
                const auto &w = regions[1];

                assert(task->regions.size() == 2);
                const auto &v_req = task->regions[0];
                const auto &w_req = task->regions[1];

                assert(v_req.privilege_fields.size() == 1);
                const Legion::FieldID v_fid = *v_req.privilege_fields.begin();

                assert(w_req.privilege_fields.size() == 1);
                const Legion::FieldID w_fid = *w_req.privilege_fields.begin();

                Legion::FieldAccessor<
                    LEGION_READ_ONLY, T, N, Legion::coord_t,
                    Realm::AffineAccessor<T, N, Legion::coord_t>
                > v_reader{v, v_fid}, w_reader{w, w_fid};

                const Legion::Domain v_domain = rt->get_index_space_domain(
                    ctx, v_req.region.get_index_space()
                );

                const Legion::Domain w_domain = rt->get_index_space_domain(
                    ctx, w_req.region.get_index_space()
                );

                assert(v_domain == w_domain);

                T result = static_cast<T>(0);
                for (Legion::RectInDomainIterator<N> it{v_domain}; it(); ++it) {
                    const Legion::Rect<N> rect = *it;
                    T temp = static_cast<T>(0);
                    Kokkos::parallel_reduce(
                        KokkosRangeFactory<ExecutionSpace, N>::create(
                            rect, ctx, rt
                        ),
                        KokkosDotProductFunctor<ExecutionSpace, T, N>{
                            v_reader.accessor, w_reader.accessor
                        },
                        temp
                    );
                    result += temp;
                }
                return result;
            }

        }; // struct KokkosTaskBody

    }; // struct DotProductTask
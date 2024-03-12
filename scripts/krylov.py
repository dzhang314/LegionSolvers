import numpy as _np
import numpy.typing as _npt


def cg(
    A: _npt.NDArray[_np.float64],
    b: _npt.NDArray[_np.float64],
    x: _npt.NDArray[_np.float64],
):
    n: int = len(x)
    assert (n, n) == A.shape
    assert (n,) == b.shape
    assert (n,) == x.shape

    # workspace: 1 scalar, 3 vectors
    r_norm2 = 0.0
    p = _np.empty(n)
    q = _np.empty(n)
    r = _np.empty(n)

    _np.copyto(r, b)
    r -= A @ x
    r_norm2 = r @ r
    _np.copyto(p, r)

    iterates: list[_npt.NDArray[_np.float64]] = []

    for _ in range(n):
        _np.matmul(A, p, out=q)
        alpha = r_norm2 / (p @ q)
        x += alpha * p
        iterates.append(_np.copy(x))
        r -= alpha * q
        r_norm2_old = r_norm2
        r_norm2 = r @ r
        beta = r_norm2 / r_norm2_old
        p *= beta
        p += r

    return iterates


def double_cg(
    A: _npt.NDArray[_np.float64],
    b: _npt.NDArray[_np.float64],
    x: _npt.NDArray[_np.float64],
):
    n: int = len(x)
    assert (n, n) == A.shape
    assert (n,) == b.shape
    assert (n,) == x.shape

    # workspace: 1 scalar, 5 vectors
    r_norm2 = 0.0
    p = _np.empty(n)
    q = _np.empty(n)
    r = _np.empty(n)
    s = _np.empty(n)
    t = _np.empty(n)

    _np.copyto(r, b)
    r -= A @ x
    r_norm2 = r @ r
    _np.copyto(p, r)

    iterates: list[_npt.NDArray[_np.float64]] = []

    for _ in range((n + 1) // 2):
        _np.matmul(A, p, out=q)
        _np.matmul(A, r, out=s)
        _np.matmul(A, q, out=t)
        alpha = r_norm2 / (p @ q)
        x += alpha * p
        iterates.append(_np.copy(x))
        r -= alpha * q
        r_norm2_old = r_norm2
        r_norm2 = r @ r
        beta = r_norm2 / r_norm2_old
        p *= beta
        p += r
        q *= beta
        q += s
        q -= alpha * t
        alpha_next = r_norm2 / (p @ q)
        x += alpha_next * p
        iterates.append(_np.copy(x))
        r -= alpha_next * q
        r_norm2_next = r @ r
        beta_next = r_norm2_next / r_norm2
        p = r + beta_next * p
        r_norm2 = r_norm2_next

    return iterates

import numpy as np
from typing import Callable, Literal
from numpy.typing import NDArray, ArrayLike

# RHS(t, y) -> dy/dt as 1D array
FloatArray = NDArray[np.floating]
RHS        = Callable[[float, FloatArray], FloatArray]
Residual   = Callable[[FloatArray], FloatArray]
Jacobian   = Callable[[FloatArray], FloatArray]

def num_jacobian(fun_y: Callable[[FloatArray], FloatArray], x: FloatArray) -> FloatArray:
    """Numerical Jacobian J = d/dx fun_y(x)."""
    x  = np.asarray(x, dtype=float)
    n  = x.size
    J  = np.empty((n, n), dtype=float)
    fx = fun_y(x)
    eps = np.finfo(float).eps
    hvec = np.sqrt(eps) * np.maximum(1.0, np.abs(x))
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = hvec[j]
        J[:, j] = (fun_y(x + e) - fx) / hvec[j]
    return J


def newton_it(
    G:  Residual,
    x0: FloatArray,
    *,
    tol: float = 1e-10,
    max_it: int = 8,
    ) -> FloatArray:
    """Solve G(x)=0 using Newton's method with numerical Jacobian."""
    x = np.asarray(x0, dtype=float)

    for it in range(max_it):
        g = G(x)
        J  = num_jacobian(G, x)
        dx = np.linalg.solve(J, -g)
        x  = x + dx
        if np.linalg.norm(dx, np.inf) <= tol * (1.0 + float(np.linalg.norm(x, np.inf))):
            return x
    return x

# explicit/forward Euler
def ee(
    f: RHS,
    t_span: tuple[float, float],
    y0: ArrayLike,
    h: float,
    ) -> tuple[FloatArray, FloatArray]:
    """
    Explicit Euler integrator.

    Parameters
    ----------
    f      : callable - RHS with signature f(t, y) -> dy/dt.
    t_span : (t0, tf) - Start and end time. Assumes uniform fixed step h.
    y0     : array-like - Initial state (1D).
    h      : float - Fixed step size (must be > 0).

    Returns
    ----------
    t      : ndarray, shape (n,) - Time grid t0, t0+h, ..., t0+(n-1)*h, ..., tf.
    y      : ndarray, shape (n, m) - State history; y[i] corresponds to time t[i].
    """
    t0, tf = t_span

    n: int = int(np.floor((tf - t0) / h)) + 1

    t: FloatArray = np.linspace(t0, t0 + (n - 1) * h, n, dtype=float)

    y0: FloatArray = np.array(y0, dtype=float).reshape(-1)
    m: int = y0.size

    y: FloatArray = np.empty((n, m), dtype=float)
    y[0, :] = y0

    for k in range(n-1):
        tk: float = float(t[k])  # type-cast to avoid warnings
        yk = y[k]

        y[k+1] = yk + h * f(tk, yk)

    return t, y


# implicit/backward Euler
def ie(
    f: RHS,
    t_span: tuple[float, float],
    y0: ArrayLike,
    h: float,
    ) -> tuple[FloatArray, FloatArray]:
    """
    Implicit Euler integrator.

    Parameters
    ----------
    f      : callable - RHS with signature f(t, y) -> dy/dt.
    t_span : (t0, tf) - Start and end time. Assumes uniform fixed step h.
    y0     : array-like - Initial state (1D).
    h      : float - Fixed step size (must be > 0).

    Returns
    ----------
    t      : ndarray, shape (n,) - Time grid t0, t0+h, ..., t0+(n-1)*h, ..., tf.
    y      : ndarray, shape (n, m) - State history; y[i] corresponds to time t[i].
    """
    t0, tf = t_span

    n: int = int(np.floor((tf - t0) / h)) + 1

    t: FloatArray = np.linspace(t0, t0 + (n - 1) * h, n, dtype=float)

    y0: FloatArray = np.array(y0, dtype=float).reshape(-1)
    m: int = y0.size

    y: FloatArray = np.empty((n, m), dtype=float)
    y[0, :] = y0

    for k in range(n - 1):
        tk:   float = float(t[k])  # type-cast to avoid warnings
        tkp1: float = float(t[k+1])
        yk = y[k]

        fk = f(tk, yk)
        x = yk + h*fk # Euler predictor

        def G(x: FloatArray) -> FloatArray:
            return x - yk - h * f(tkp1, x)

        x = newton_it(G, x)

        # the implicit Euler update is just the converged x
        y[k + 1] = x

    return t, y

# trapezoidal rule
def trap(
    f: RHS,
    t_span: tuple[float, float],
    y0: ArrayLike,
    h: float,
    ) -> tuple[FloatArray, FloatArray]:
    """
    Trapezoidal integrator.

    Parameters
    ----------
    f      : callable - RHS with signature f(t, y) -> dy/dt.
    t_span : (t0, tf) - Start and end time. Assumes uniform fixed step h.
    y0     : array-like - Initial state (1D).
    h      : float - Fixed step size (must be > 0).

    Returns
    ----------
    t      : ndarray, shape (n,) - Time grid t0, t0+h, ..., t0+(n-1)*h, ..., tf.
    y      : ndarray, shape (n, m) - State history; y[i] corresponds to time t[i].
    """
    t0, tf = t_span

    n: int = int(np.floor((tf - t0) / h)) + 1

    t: FloatArray = np.linspace(t0, t0 + (n - 1) * h, n, dtype=float)

    y0: FloatArray = np.array(y0, dtype=float).reshape(-1)
    m: int = y0.size

    y: FloatArray = np.empty((n, m), dtype=float)
    y[0, :] = y0

    for k in range(n - 1):
        tk: float = float(t[k])  # type-cast to avoid warnings
        tkp1: float = float(t[k + 1])
        yk = y[k]

        fk = f(tk, yk)
        x = yk + h * fk  # Euler predictor

        def G(x: FloatArray) -> FloatArray:
            return x - yk - 1/2 * h * (fk + f(tkp1, x))

        x = newton_it(G, x)

        # the trapezoidal update is just the converged x
        y[k + 1] = x

    return t, y



# standard 4th-order Runge-Kutta
def rk4(
    f: RHS,
    t_span: tuple[float, float],
    y0: ArrayLike,
    h: float,
    ) -> tuple[FloatArray, FloatArray]:
    """
    Standard 4th-order Runge–Kutta integrator.

    Parameters
    ----------
    f      : callable - RHS with signature f(t, y) -> dy/dt.
    t_span : (t0, tf) - Start and end time. Assumes uniform fixed step h.
    y0     : array-like - Initial state (1D).
    h      : float - Fixed step size (must be > 0).

    Returns
    ----------
    t      : ndarray, shape (n,) - Time grid t0, t0+h, ..., t0+(n-1)h up to (but not exceeding) tf.
    y      : ndarray, shape (n, m) - State history; y[i] corresponds to time t[i].
    """
    t0, tf = t_span

    n: int = int(np.floor((tf - t0) / h)) + 1
    t: FloatArray = np.linspace(t0, t0 + (n - 1) * h, n, dtype=float)

    y0: FloatArray = np.array(y0, dtype=float).reshape(-1)
    m: int = y0.size

    y: FloatArray = np.empty((n, m), dtype=float)
    y[0, :] = y0

    for k in range(n - 1):
        tk: float = float(t[k]) # type-cast to avoid warnings
        yk = y[k]

        k1 = f(tk,           yk)
        k2 = f(tk + 0.5 * h, yk + 0.5 * h * k1)
        k3 = f(tk + 0.5 * h, yk + 0.5 * h * k2)
        k4 = f(tk + h,       yk + h * k3)

        y[k + 1] = yk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t, y

def rk45(
    f: RHS,
    t_span: tuple[float, float],
    y0: ArrayLike,
    h : float,
    *,
    abstol: float | FloatArray = 1e-6,
    reltol: float = 1e-3,
    safety: float = 0.9,
    min_step: float = 1e-12,
    # max_step: Optional[float] = None,
    order: Literal[4, 5] = 4,
    ) -> tuple[FloatArray, FloatArray]:
    """
    Runge–Kutta–Fehlberg 4(5) with adaptive step size.

    Sharp tolerances: a step is acceptable only if BOTH per-component
    absolute and relative tests pass. Controller tries one growth step,
    then shrinks in a loop until acceptable (or min_step reached).

    Parameters
    ----------
    f        : RHS - Right-hand side function f(t, y)->dy/dt (1D array).
    t_span   : (t0, tf) - Start and end times (t0 < tf).
    y0       : array-like - Initial state (1D).
    h        : float - Initial h guess; default (tf - t0)/100.
    abstol   : float or array - Absolute tolerance (per component if array).
    reltol   : float - Relative tolerance (scalar).
    safety   : float - Safety factor in (0,1), default 0.9.
    min_step : float - Minimum step size; if reached and tolerances still fail, a warning-like message is printed once and the step is accepted.
    order    : 4 or 5 - Which member of the embedded pair to propagate (y4 or y5).

    Returns
    ----------
    t        : array, shape (N,) -  Monotone time grid ending at tf.
    Y        : array, shape (N, ny) - States (each row corresponds to t[i]).
    """
    t0, tf = t_span

    y = np.asarray(y0, dtype=float).reshape(-1)
    ny: int = y.size

    # tolerances
    if np.ndim(abstol) == 0:
        absvec = abstol * np.ones(ny, dtype=float)
    else:
        absvec = np.asarray(abstol, dtype=float).reshape(-1)
        if absvec.size != ny:
            raise ValueError("abstol size must match y0 dimension.")
    rel  = float(reltol)
    sfty = float(safety)

    # initial step and bounds
    if min_step <= 0.0:
        raise ValueError("min_step must be > 0.")
    if h <= 0.0:
        h = min_step
    h = max(h, min_step)

    # storage (growable)
    cap = 1024
    t = np.empty(cap, dtype=float)
    Y = np.empty((cap, ny), dtype=float)
    k = 0
    t[k] = t0
    Y[k] = y

    # integration state
    tn = t0
    yn = y.copy()
    pord = 4  # lower order in the 4(5) pair
    min_step_warned = False

    eps = np.finfo(float).eps

    while tn < tf:
        # clamp step to hit tf; allow last step < min_step
        if tf - tn <= min_step:
            h = tf - tn
        else:
            h = min(h, tf - tn)
            h = max(h, min_step)

        # one trial with current h
        y4, y5 = _rkpair_fehlberg(f, tn, yn, h)

        y_keep = y4 if order == 4 else y5

        # sharp tolerances (per component)
        err_vec = np.abs(y5 - y4)
        M = np.abs(y5)
        # avoid divide-by-zero
        errA = np.max(err_vec / np.maximum(absvec, eps))
        errR = np.max(err_vec / np.maximum(rel * M, eps))
        err = max(errA, errR)
        if err == 0.0:
            err = 1e-16  # avoid divide-by-zero in fac

        # propose one growth attempt (best-by-test)
        fac_grow = sfty * err ** (-1.0 / (pord + 1.0))  # exponent = 1/5
        h_try = h * fac_grow
        h_try = min(h_try, tf - tn)
        h_try = max(h_try, min_step)

        # only re-try if meaningfully increased
        if h_try > h * (1.0 + (1.0 - sfty) / 100.0):
            h = h_try
            y4, y5 = _rkpair_fehlberg(f, tn, yn, h)
            y_keep = y4 if order == 4 else y5
            err_vec = np.abs(y5 - y4)
            M = np.abs(y5)
            errA = np.max(err_vec / np.maximum(absvec, eps))
            errR = np.max(err_vec / np.maximum(rel * M, eps))
            err = max(errA, errR)
            if err == 0.0:
                err = 1e-16

        # shrink until acceptable (err <= 1)
        while err > 1.0:
            fac_grow = sfty * err ** (-1.0 / (pord + 1.0))
            h_new = h * fac_grow
            h_new = min(h_new, tf - tn)
            h_new = max(h_new, min_step)

            # stuck? (usually at min_step)
            if abs(h_new - h) <= eps:
                if not min_step_warned:
                    print(
                        f"[rkf45] MinStep (h={min_step:g}) prevents meeting tolerances at t={tn:.6g}; "
                        "proceeding anyway."
                    )
                    min_step_warned = True
                break

            # try smaller step
            h = h_new
            y4, y5 = _rkpair_fehlberg(f, tn, yn, h)
            y_keep = y4 if order == 4 else y5
            err_vec = np.abs(y5 - y4)
            M = np.abs(y5)
            errA = np.max(err_vec / np.maximum(absvec, eps))
            errR = np.max(err_vec / np.maximum(rel * M, eps))
            err = max(errA, errR)
            if err == 0.0:
                err = 1e-16

        # accept step
        tn = tn + h
        yn = y_keep

        # store (grow if needed)
        k += 1
        if k >= cap:
            new_cap = int(round(1.5 * cap))
            t = np.resize(t, new_cap)
            Y = np.resize(Y, (new_cap, ny))
            cap = new_cap

        t[k] = tn
        Y[k] = yn

    # trim
    return t[: k + 1].copy(), Y[: k + 1].copy()


def _rkpair_fehlberg(
    f: RHS, t: float, y: FloatArray, h: float
    ) -> tuple[FloatArray, FloatArray]:
    """Fehlberg 4(5) embedded RK pair: returns (y4, y5)."""
    # Coefficients
    c2, c3, c4, c5, c6 = 1/4, 3/8, 12/13, 1.0, 1/2
    a21 = 1/4
    a31, a32 = 3/32, 9/32
    a41, a42, a43 = 1932/2197, -7200/2197, 7296/2197
    a51, a52, a53, a54 = 439/216, -8.0, 3680/513, -845/4104
    a61, a62, a63, a64, a65 = -8/27, 2.0, -3544/2565, 1859/4104, -11/40

    b4 = np.array([25/216, 0.0, 1408/2565, 2197/4104, -1/5, 0.0], dtype=float)
    b5 = np.array([16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55], dtype=float)

    k1 = f(t,               y)
    k2 = f(t + c2*h,        y + h*(a21*k1))
    k3 = f(t + c3*h,        y + h*(a31*k1 + a32*k2))
    k4 = f(t + c4*h,        y + h*(a41*k1 + a42*k2 + a43*k3))
    k5 = f(t + c5*h,        y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
    k6 = f(t + c6*h,        y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

    # y4, y5
    K = np.column_stack((k1, k2, k3, k4, k5, k6))  # shape (m,6)
    y4 = y + h * (K @ b4)
    y5 = y + h * (K @ b5)
    return y4, y5
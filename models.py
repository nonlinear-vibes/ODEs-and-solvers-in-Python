"""ODE right-hand sides (RHS) and labels for plotting."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]

# Labels used for legends
labels: dict[str, tuple[str, ...]] = {
    "vdp":(r"$x$", r"$\dot{x}$"),
    "lv":(r"$x$", r"$y$"),
    "rayleigh":(r"$y$", r"$\dot{y}$"),
    "coupled_pendulums":(r"$\phi_1$", r"$\dot{\phi}_1$", r"$\phi_2$", r"$\dot{\phi}_2$", r"$q$", r"$\dot{q}$"),
    "cr3bp":(r"$x$", r"$\dot{x}$", r"$y$", r"$\dot{y}$"),
    "sir":(r"$S$", r"$I$", r"$R$"),
}

# -------------------------------------
# van der Pol oscillator
def vdp(t: float, state: FloatArray, *, mu: float = 3.0) -> FloatArray:
    """RHS for the van der Pol oscillator.

    Parameters
    ----------
    t     : float - Time (unused, included for solver signature).
    state : Array, shape (2,) - State vector [x, xdot].
    mu    : float - Nonlinearity parameter.

    Returns
    ----------
    [xdot, xddot] : FloatArray, shape (2,) - Time derivatives.
    """
    if state.shape[0] != 2:
        raise ValueError(f"vdp expects state of length 2, got shape {state.shape}")
    x, xdot = state
    return np.array([xdot, mu * (1 - x**2) * xdot - x], dtype=float)

# -------------------------------------
# Lotka-Volterra system
def lv(
    t: float,
    state: FloatArray,
    *,
    a: float = 2/3,
    b: float = 4/3,
    c: float = 1,
    d: float = 1
    ) -> FloatArray:
    """RHS for the Lotka-Volterra system.

    Parameters
    ----------
    t     : float - Time (unused, included for solver signature).
    state : Array, shape (2,) - State vector [x, y].
    a, b, c, d : float - System parameters.

    Returns
    ----------
    [xdot, ydot] : FloatArray, shape (2,) - Rate of change of predators and preys.
    """
    if state.shape[0] != 2:
        raise ValueError(f"lv expects state of length 2, got shape {state.shape}")
    x, y = state
    return np.array([a * x - b * x * y, c * x * y - d * y], dtype=float)

# -------------------------------------
# Rayleigh system
def rayleigh(
    t: float,
    state: FloatArray,
    *,
    mu: float = 3.0,
    ) -> FloatArray:
    """RHS for the Rayleigh oscillator (nonlinear damping on y').

    Parameters
    ----------
    t     : float - Time (unused, included for solver signature).
    state : Array, shape (2,) - State vector [y, ydot].
    mu    : float - Nonlinear damping parameter.

    Returns
    ----------
    [ydot, yddot] : FloatArray, shape (2,) - Time derivatives.
    """
    if state.shape[0] != 2:
        raise ValueError(f"rayleigh expects state of length 2, got shape {state.shape}")
    y, ydot = state
    return np.array([ydot, mu * (1 - ydot**2)*ydot - y], dtype=float)

# -------------------------------------
# Two pendulums on a cart:
def coupled_pendulums(
    t: float,
    state: FloatArray,
    *,
    m1: float = 0.1,
    m2: float = 0.1,
    m_cart: float = 5.0,
    l: float = 0.2,
    g: float = 9.81,
    c_cart: float = 0.2,
    c_piv: float = 0.0,
    ) -> FloatArray:
    """RHS for two pendulums mounted on a translating cart.

    Parameters
    ----------
    t     : float - Time (unused, included for solver signature).
    state : Array, shape (6,) - [phi1, dphi1, phi2, dphi2, q, dq].
    m1, m2, m_cart, l, g, c_cart, c_piv : float - Physical parameters (keyword-only).

    Returns
    ----------
    [dphi1, ddphi1, dphi2, ddphi2, dq, ddq] : Array, shape (6,)
    """
    if state.shape[0] != 6:
        raise ValueError(f"coupled_pendulums expects y of length 6, got shape {state.shape}")

    phi1, dphi1, phi2, dphi2, q, dq = state

    alpha1 = m1 * l / (m_cart + m1 + m2)
    alpha2 = m2 * l / (m_cart + m1 + m2)

    denom = 1 - (np.cos(phi1) ** 2) * (alpha1 / l) - (np.cos(phi2) ** 2) * (alpha2 / l)

    # numeric guard
    if abs(denom) < 1e-12:
        denom = 1e-912 if denom >= 0 else -1e-12

    num_core = (alpha1 * (np.sin(phi1) * dphi1 ** 2 + (g / l) * np.cos(phi1) * np.sin(phi1))
              + alpha2 * (np.sin(phi2) * dphi2 ** 2 + (g / l) * np.cos(phi2) * np.sin(phi2)))

    cart_damp = -(c_cart / (m_cart + m1 + m2)) * dq
    piv_damp  = ((c_piv / (m1 * l**2)) * alpha1 * np.cos(phi1) * dphi1
               + (c_piv / (m2 * l**2)) * alpha2 * np.cos(phi2) * dphi2)

    ddq = (num_core + cart_damp + piv_damp) / denom

    ddphi1 = -(g / l) * np.sin(phi1) - (ddq / l) * np.cos(phi1) - (c_piv / (m1 * l**2)) * dphi1
    ddphi2 = -(g / l) * np.sin(phi2) - (ddq / l) * np.cos(phi2) - (c_piv / (m2 * l**2)) * dphi2

    return np.array([dphi1, ddphi1, dphi2, ddphi2, dq, ddq], dtype=float)


# -------------------------------------
# Circular restricted three-body problem:
def cr3bp(
    t: float,
    state: FloatArray,
    *,
    mu: float = 0.012277471,
    ) -> FloatArray:
    """RHS for the circular restricted three-body problem.

    Parameters
    ----------
    t     : float - Time (unused, included for solver signature).
    state : Array, shape (4,) - [x, Xdot, y, ydot] in the rotating frame (angular speed = 1).
    mu    : float - Mass parameter.

    Returns
    ----------
    [xdot, xddot, ydot, yddot] : FloatArray, shape (4,)
    """
    if state.shape[0] != 4:
        raise ValueError(f"cr3bp expects state of length 4, got shape {state.shape}")

    x, xdot, y, ydot = state

    nu = 1 - mu

    # distances to the primaries
    r1sq = (x + mu)**2 + y**2   # to (-mu, 0)
    r2sq = (x - nu)**2 + y**2   # to (+nu, 0)
    r1_3 = r1sq**(3/2)
    r2_3 = r2sq**(3/2)

    # dynamics in rotating frame
    xddot = x + 2*ydot - nu*(x + mu)/r1_3 - mu*(x - nu)/r2_3
    yddot = y - 2*xdot - nu * y/r1_3      - mu*y/r2_3

    return np.array([xdot, xddot, ydot, yddot], dtype=float)

def sir(
    t: float,
    state: FloatArray,
    *,
    beta: float = 0.30,   # infection rate
    gamma: float = 0.10,  # recovery rate
    ) -> FloatArray:
    """RHS of the simple SIR model.

    Parameters
    ----------
    t     : float - Time (unused, included for solver signature).
    state : FloatArray, shape (3,) - [S, I, R]

    Returns
    ----------
    [S', I', R'] : FloatArray, shape (3,)

    Equations:
        S' = -beta * S * I / N
        I' =  beta * S * I / N - gamma * I
        R' =  gamma * I
    """
    if state.shape[0] != 3:
        raise ValueError(f"sir expects state of length 3, got shape {state.shape}")

    S, I, R = state

    lambda_ = beta * S * I
    Sdot = -lambda_
    Idot =  lambda_ - gamma * I
    Rdot =  gamma * I

    return np.array([Sdot, Idot, Rdot], dtype=float)
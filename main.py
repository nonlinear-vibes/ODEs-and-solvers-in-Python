import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Callable, TypeAlias
from functools import partial

import models
import solvers

RHS:    TypeAlias = Callable[[float, np.ndarray], np.ndarray]
Solver: TypeAlias = Callable[[RHS, tuple[float, float], npt.ArrayLike, float], tuple[np.ndarray, np.ndarray]]

def main():

    # change settings here
    system: str = "sir"
    solver: str = "ie"
    plot_states: tuple[int, ...] = (0, 1, 2)     # plot states against time
    plot_pairs: tuple[tuple[int, int], ...] = () # plot states against each other
    params = dict(
        beta=0.40,
        gamma=0.05
    )

    t_span: tuple[float, float] = (0.0, 100.0)
    x0: list[float] = [0.99, 0.01, 0.0]
    h: float = 0.05

    # solver, don't change anything here
    labels: tuple[str, ...] = models.labels[system]
    base_func = getattr(models, system)
    func: RHS = partial(base_func, **params)
    solve: Solver = getattr(solvers, solver)

    t, y = solve(func, t_span, x0, h)

    # Plot time-series (states vs. time)
    if plot_states:
        plt.figure()
        for state in plot_states:
            plt.plot(t, y[:, state], label=labels[state])
        plt.xlabel("t")
        if len(plot_states) == 1:
            plt.ylabel(labels[plot_states[0]])
        else:
            plt.ylabel("states")
        plt.title(f"{system}: time-series")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # Plot phase-space (state pairs)
    if plot_pairs:
        for i, j in plot_pairs:
            plt.figure()
            plt.plot(y[:, i], y[:, j], label=f"{labels[i]} vs {labels[j]}")
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
            plt.title(f"{system}: {labels[i]} vs {labels[j]}")
            plt.grid(True)
            plt.axis('equal')
            plt.tight_layout()

    plt.show()

if __name__ == '__main__':

    main()

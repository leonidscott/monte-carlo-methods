from functional import seq, pseq
from typing import Tuple, List

from math import sin, cos, pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np

import matplotlib.pyplot as plt

def numerical_model(xi: Tuple[float,float], N: int) -> Tuple[List[float], List[float]]:
    [xi1,xi2] = xi
    # 0. Construct Discretization
    L = 1 # Length of domain
    dx = L/N
    def i2x(idx: int) -> float: # Index to x coordinate
        return float(idx)*dx + 0.5*dx

    # 1. Construct A Matrix (Sparse)
    def alpha(x: float) -> float:
        return 1.0 + 0.3*xi1*sin(pi*x) + 0.2*xi2*cos(2*pi*x)

    main = (seq(range(N))
             .map(lambda i: alpha(i2x(i)+0.5*dx) + alpha(i2x(i)-0.5*dx))
             .to_list())
    sub   = (seq(range(N-1))
             .map(lambda i: i+1)
             .map(lambda i: -1* alpha(i2x(i)+0.5*dx))
             .to_list())
    upper = (seq(range(N-1))
             .map(lambda i: -1 * alpha(i2x(i)-0.5*dx))
             .to_list())
    A = diags(
        diagonals=[sub, main, upper],
        offsets=[-1, 0, 1],
        format="csr"
    )

    # 2. Construct dx^2 matrix
    rhs = np.full(N, dx**2)

    # 3. Solve sytem
    xs = list(map(i2x ,range(N)))
    u = spsolve(A,rhs)

    return xs, u
#fn done

if __name__ == "__main__":
    [x,u] = numerical_model([0.99, 0.001], 100000)
    plt.figure()
    plt.plot(x, u)
    plt.xlabel("x")
    plt.xlim(0,1)
    plt.ylabel("u")
    plt.title("Poisson Eq")
    plt.show()

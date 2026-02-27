from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt

# Local Deps
def set_cwd():
    import sys
    import os
    sys.path.insert(1, os.path.dirname(__file__))
set_cwd()
import mc_methods as mc
from mc_methods import scalar, vector
import sampler
from sampler import sampler_B
import analytic_model as plots
# Local Deps

Ns = [11,101,1001,10001,100001]

def model(xi: Tuple[float, float], N):
    import numerical_model as nummod #For parallel
    [xs, u] = nummod.numerical_model(xi, N)
    idx = xs.index(0.5)
    return u[idx]

hf = partial(model, N=10001)   #High Fidelity Function
midf = partial(model, N=1001) #Mid Fidelity Function
lf = partial(model, N=101)   #Low Fidelity Function

if __name__ == "__main__":
    # Problem 1: ========================================
    # A) Plot exp, var, rmse as N increases
    #def run_mc(N):
    #    [exp,var] = mc.monte_carlo(hf, sampler_B, 1000)
    #    return {"exp" : exp, "var" : var, "N": N, "rmse": plots.rmse(var,N)}
    #mc_metrics = list(map(run_mc, Ns))
    #plots.plot_mc_metrics(mc_metrics, "Convergence of MC methods")

    # B) KDE Plot
    #[pdf_in, pdf_out] = mc.mykde(hf, sampler_B, 1000)
    #plots.plot_kde(pdf_in, pdf_out)

    # A) Plot RMSE as N incrases
    def mc_rmse(N,M):
        def rmc(M):
            [exp, _] = mc.monte_carlo(hf, sampler_B, N)
            print({'N':N, 'M':M, 'exp':exp})
            return exp
        exps = list(map(rmc, range(M)))
        return exps
    mc_rmses = list(map(partial(mc_rmse, M=50), Ns))
    plt.show()

from pprint import pprint
from operator import itemgetter
from functools import partial
from functional import seq
from typing import Tuple
import os

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

Ns = [10,100,1000,10000,100000]

@scalar
def model(xi: Tuple[float, float], N):
    import numerical_model as nummod #For parallel
    [xs, u] = nummod.numerical_model(xi, N)
    idx = xs.index(0.5)
    return u[idx]

hf = partial(model, N=1001)   #High Fidelity Function
midf = partial(model, N=101)  #Mid Fidelity Function
lf = partial(model, N=11)     #Low Fidelity Function
hf.kind = "scalar"
midf.kind = "scalar"
lf.kind = "scalar"

if __name__ == "__main__":
    print(os.cpu_count())
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

    # C) Plot RMSE as N incrases
    def mc_rmse(N,M):
        def rmc(M):
            [exp, var] = mc.monte_carlo(hf, sampler_B, N)
            momes = {'N':N, 'M':M, 'exp':exp, 'var':var, 'rmse':plots.rmse(var,N)}
            print(momes)
            return momes
        exps = list(map(rmc, range(M)))
        return exps
    mc_momes= list(map(partial(mc_rmse, M=10), Ns))

    # clean data for plot
    mc_momes_by_m = (
        seq(mc_momes)
        .flatten()                                # flatten nested lists
        .group_by(itemgetter("M"))                # group by M
        .map(lambda gby : gby[1])                 # discard keys (M)
        .map(lambda group:
             seq(group)
             .sorted(key=itemgetter("N"))         # sort each group by N
             .to_list()
        )
        .sorted(key=lambda group: group[0]["M"])  # sort groups by M
        .to_list()
    )
    print("")
    pprint(mc_momes_by_m)
    plots.plot_many_metrics(mc_momes_by_m, "Numberical Model")

    # Problem 2: =======================================
    #print("Control Variates")
    #def cv_rmse(N,M):
    #    def rcv(M):
    #        [exp, _, alpha] = mc.control_variate(lf, hf, N*100+1, N+1, sampler_B)
    #        print({'N':N, 'M':M, 'exp':exp, 'Nlow' : N*100, 'Nhigh' : N, 'alpha':alpha})
    #        return exp
    #    exps = list(map(rcv, range(M)))
    #    return exps
    #cv_rmses = list(map(partial(cv_rmse, M=50), Ns))

    # Problem 3: =======================================
    #print("Multi Level Monte Carlo")

    #fns = [lf,midf,hf]
    #def mlmc(N,M):
    #    def run_mlmc(M):
    #        Nvec = [N*100+1, N*10+1, N+1]
    #        exp = mc.multi_level(fns, Nvec,sampler_B)
    #        print({'N':N, 'M':M, 'exp':exp, 'N_vec': Nvec})
    #        return exp
    #    exps = list(map(run_mlmc, range(M)))
    #    return exps
    #cv_rmses = list(map(partial(mlmc, M=50), [100000]))
    plt.show()


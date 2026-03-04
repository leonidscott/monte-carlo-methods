from pprint import pprint
from operator import itemgetter
from functional import seq,pseq
from functools import partial
from typing import Tuple, List, TypedDict, TypedDict, cast, Any, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math
import mc_methods
import numpy as np
import numpy.fft as fft
from scipy import fft as sfft
import os

# Local Deps
def set_cwd():
    import sys
    import os
    sys.path.insert(1, os.path.dirname(__file__))
set_cwd()
import mc_methods as mc
from mc_methods import scalar, vector
# Local Deps

# Problem 1:
@scalar
def analytical_model(X: Tuple[float, float, float],
                     theta: Tuple[float, float]) -> float:
    import math #For Parallel Execution (LOL Python is stupid)
    [alpha, beta] = theta
    return math.log(1.0 + math.exp(alpha*(X[0] + X[2]))) * math.exp(beta * X[1]**2)

def rmse(var, N):
    return var/math.sqrt(N)

class MC_Metric(TypedDict):
    N: int
    exp: float
    var: float
    rmse: float

def get_all(d, key):
    return list(map(lambda met: met[key], d))

def plot_mc_metrics(mc_metrics: List[dict], title) -> None:
    # Lol no better way to do destructuring on a dictionary grrr
    list(map(lambda el: cast(MC_Metric, el), mc_metrics))
    def get_all(key):
        return list(map(lambda met: met[key], mc_metrics))
    Ns = get_all("N")

    plt.figure()
    plt.plot(Ns, get_all("exp"), label="Expectation")
    plt.plot(Ns, get_all("var"), label="Variance")
    plt.plot(Ns, get_all("rmse"), label="RMSE")
    plt.xlabel("N")
    plt.ylabel("Metric")
    plt.yscale("log")
    plt.legend()
    plt.title(title)

def plot_many_metrics(runs, title) -> None:
    plt.figure()

    # Generate distinct colors for each run (each M)
    colors = cm.tab10(np.linspace(0, 1, len(runs)))

    for i, run in enumerate(runs):
        Ns = get_all(run, 'N')

        # Only label metrics once (for legend clarity)
        label_exp  = "Expectation" if i == 0 else None
        label_var  = "Variance"    if i == 0 else None
        label_rmse = "RMSE"        if i == 0 else None

        plt.plot(Ns, get_all(run, "exp"), color=colors[i], linestyle='-', label=label_exp)
        plt.plot(Ns, get_all(run, "var"), color=colors[i], linestyle='--', label=label_var)
        plt.plot(Ns, get_all(run, "rmse"), color=colors[i], linestyle=':', label=label_rmse)
    plt.xlabel("N")
    plt.ylabel("Metric")
    plt.yscale("log")
    plt.legend()
    plt.title(title)

def plot_kde(f_vals,kde_vals,title="KDE"):
    plt.figure()
    plt.plot(f_vals, kde_vals)
    plt.xlabel("f(x)")
    plt.ylabel("KDE: p(f(x))")
    plt.title(title)

def plot_rmse(title, Ns, *rmses):
    plt.figure()
    for rmse in rmses:
        plt.plot(Ns, rmse['rmse'], label=rmse['label'])
    plt.xlabel("N")
    plt.ylabel("RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(title)


# Problem 2:
@vector
def truncate_model(evals: List[float], K: int) -> List[float]:
    length = len(evals)
    evals = np.asarray(evals, dtype=np.float32)

    # Fourier Transform + Truncation
    fevals = sfft.rfft(evals, workers=-1)

    trunc = np.zeros_like(fevals)
    trunc[:K+1] = fevals[:K+1]
    # Inverse Transform
    return sfft.irfft(trunc, n=length, workers=-1)


if __name__ == "__main__":
    # Problem 1: ========================================
    print(os.cpu_count())

    def sampler() -> Tuple[float, float, float]:
        """Parallel-safe normal sampler.

        Each process lazily initializes its own RNG using a SeedSequence that
        mixes a base seed with the process id, so forked workers do not reuse
        identical RNG state.
        """
        import os
        import numpy as np
        # HACK: Add base seed and sampler rng to a utils module
        from mc_methods import BASE_SEED, SAMPLER_RNG

        pid = os.getpid()
        if SAMPLER_RNG["pid"] != pid or SAMPLER_RNG["rng"] is None:
            seed_seq = np.random.SeedSequence([BASE_SEED, pid])
            SAMPLER_RNG["pid"] = pid
            SAMPLER_RNG["rng"] = np.random.default_rng(seed_seq)

        draws = SAMPLER_RNG["rng"].normal(loc=0.0, scale=1.0, size=3)
        return float(draws[0]), float(draws[1]), float(draws[2])
    model = partial(analytical_model, theta=(2.0,-0.3))

    # A) Plot exp, var, rmse as N increases
    #print("MC Plain")
    #Ns = [10,100,1000,10000,100000]
    #def mc_rmse(N,M):
    #    def rmc(M):
    #        [exp, var] = mc.monte_carlo(model, sampler, N)
    #        momes = {'N':N, 'M':M, 'exp':exp,'var': var, 'rmse' : rmse(var,N)}
    #        print(momes)
    #        return momes
    #    exps = list(map(rmc, range(M)))
    #    return exps
    #mc_momes= list(map(partial(mc_rmse, M=10), Ns))

    ## clean data for plot
    #mc_momes_by_m = (
    #    seq(mc_momes)
    #    .flatten()                                # flatten nested lists
    #    .group_by(itemgetter("M"))                # group by M
    #    .map(lambda gby : gby[1])                 # discard keys (M)
    #    .map(lambda group:
    #         seq(group)
    #         .sorted(key=itemgetter("N"))         # sort each group by N
    #         .to_list()
    #    )
    #    .sorted(key=lambda group: group[0]["M"])  # sort groups by M
    #    .to_list()
    #)
    #print("")
    #pprint(mc_momes_by_m)
    #plot_many_metrics(mc_momes_by_m, "Analytical Model - Beta = -0.3")

    # KDE Plot
    #[pdf_in, pdf_out] = mc.kde(model, sampler, 1000)
    #[pdf_in2, pdf_out2] = mc.mykde(model, sampler, 1000)
    #plot_kde(pdf_in, pdf_out)
    #plot_kde(pdf_in2, pdf_out2, "KDE Analytical Solution")

    # Problem 2: =======================================
    #x0_vals = np.linspace(-5,5,200)
    #x_vals = seq(x0_vals).map(lambda x0: [x0, 1.0, 1.0]).to_list()
    #pure_evals = pseq(x_vals).map(model).to_list()
    #truc_evals = truncate_model(pure_evals, K=15)

    #plt.figure()
    #plt.plot(x0_vals, pure_evals, label="True Model")
    #plt.plot(x0_vals, truc_evals, label="Trunc K=5")
    #plt.xlabel("x0")
    #plt.ylabel("model")
    #plt.title("Fourier Truncation of Analytical Model: K=15")

    #print("Control Variate")
    @vector
    def lf(samples: Tuple[float, float, float]) -> List[float]:
        pure_evals = pseq(samples).map(model).to_list()
        return truncate_model(pure_evals, K=5)

    model.kind = "scalar"

    #N_high= Ns
    #N_low = list(map(lambda n : n*100, N_high))
    #def cv_rmse(N,M):
    #    def rcv(M):
    #        [exp, _, alpha] = mc.control_variate(lf, model, N*100, N, sampler)
    #        print({'N':N, 'M':M, 'exp':exp, 'Nlow' : N*100, 'Nhigh' : N, 'alpha':alpha})
    #        return exp
    #    exps = list(map(rcv, range(M)))
    #    return exps
    #cv_rmses = list(map(partial(cv_rmse, M=50), N_high))

    # Problem 3: =======================================
    #print("Multi Level Monte Carlo")
    #@vector
    #def midf(samples: Tuple[float, float, float]) -> List[float]:
    #    pure_evals = pseq(samples).map(model).to_list()
    #    return truncate_model(pure_evals, K=10)

    #fns = [lf, midf, model]
    #def mlmc(N,M):
    #    def run_mlmc(M):
    #        Nvec = [N, N*10, N*100]
    #        exp = mc.multi_level(fns, Nvec,sampler)
    #        print({'N':N, 'M':M, 'exp':exp, 'N_vec': Nvec})
    #        return exp
    #    exps = list(map(run_mlmc, range(M)))
    #    return exps
    #cv_rmses = list(map(partial(mlmc, M=50), Ns))

    plt.show()

from __future__ import annotations
from typing import Tuple, Callable, TypeVar, List, Protocol, runtime_checkable, Literal
from functional import seq, pseq
# Stats Deps
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde

T = TypeVar("T")
def monte_carlo(*args) -> Tuple[float, float]:
    if len(args) == 3:
        return monte_carlo_sample(*args)
    elif len(args) == 1:
        return monte_carlo_no_sample(*args)
    else:
        raise ValueError("Wrong number of arguments for monte_carlo")

T = TypeVar("T")
def monte_carlo_sample(model: Callable[[T], float],
                       sampler: Callable[[],T],
                       N: int) -> Tuple[float, float]:
    samples = (pseq(range(N))
               .map(lambda _: model(sampler())).to_list())
    exp = 1.0/N * sum(samples)
    var = 1.0/(N-1.0) * (seq(samples)
                         .map(lambda s: (s-exp)**2)
                         .sum())
    return exp,var

T = TypeVar("T")
def monte_carlo_no_sample(samples: List[float]) -> Tuple[float, float]:
    N = len(samples)
    exp = 1.0/N * sum(samples)
    var = 1.0/(N-1.0) * (seq(samples)
                         .map(lambda s: (s-exp)**2)
                         .sum())
    return exp,var

T = TypeVar("T")
def mykde(f: Callable[[T], float],
          sampler: Callable[[], T],
          N: int) -> Tuple[List[float], List[float]]:

    # Create N Evaluations for KDE
    evals = pseq(range(N)).map(lambda _: f(sampler())).to_list()

    # Create KDE Series
    n = len(evals)
    bw = n**(-1/5) # Scott's Rule, Naturally
    dists = (pseq(evals)
             .map(lambda xi: stats.norm(loc=xi, scale=bw))
             .to_list())
    def kde_onex(x):
        return 1/(n*bw) * (seq(dists)
                           .map(lambda k : k.pdf(x))
                           .sum())

    # Evaluate the KDE function at many points to create a smooth pdf
    fs = np.linspace(min(evals), max(evals), 1000)
    kde_values = pseq(fs).map(kde_onex).to_list()
    return fs, kde_values

T = TypeVar("T")
def kde(f: Callable[[T], float],
        sampler: Callable[[], T],
        N: int) -> Tuple[List[float], List[float]]:

    #evals = list(map(lambda _: f(sampler()), range(N)))
    evals = pseq(range(N)).map(lambda _: f(sampler())).to_list()

    # KDE with manual bandwidth scaling
    kde = gaussian_kde(evals)
    kde.set_bandwidth(bw_method=0.2 / np.std(evals))

    fs = np.linspace(min(evals), max(evals), 1000)
    kde_values = kde(fs)

    return fs,kde_values

T = TypeVar("T")
def control_variate_obs(lfn: Callable[[T], float], # Low Fidelity Fn
                        hfn: Callable[[T], float], # High Fidelity Fn
                        sampler: Callable[[], T],
                        N: int) -> Tuple[List[float], List[float]]:
    # 1. Take N_low samples, build expectation and variance
    [exp_low, var_low] = monte_carlo(lowf, sampler, N_low)

    return [[5.9], [1.3]]

Tin = TypeVar("Tin", contravariant=True)
@runtime_checkable
class ScalarFn(Protocol[Tin]):
    kind: Literal["scalar"]
    def __call__(self, x: Tin) -> float: ...

Tin2 = TypeVar("Tin2")
@runtime_checkable
class VectorFn(Protocol[Tin2]):
    kind: Literal["vector"]
    def __call__(self, x: List[Tin2]) -> float: ...

Fn = ScalarFn[Tin] | VectorFn[Tin]

def scalar(f: Callable[[T], float]) -> ScalarFn:
    setattr(f, "kind", "scalar")
    return f # type: ignore[return-value]

def vector(f: Callable[[T], float]) -> VectorFn:
    setattr(f, "kind", "vector")
    return f # type: ignore[return-value]

Tcv = TypeVar("Tcv") # Tcv -> Type Control Variate
# TODO: Make sampler optional
def control_variate(lf: Fn[Tcv], hf: Fn[Tcv],
                    N_low: int, N_high: int,
                    sampler: Callable[[], Tcv]) -> Tuple[List[float], List[float]]:
    # 1. Take N_low samples to build expectation of hl
    def calc_evals(f,N) -> Tuple[List[float], List[float]]:
        #print("  -> creating samples")
        samples = (pseq(range(N)).map(lambda _: sampler()).to_list())
        #print("  -> creating evals")
        if f.kind == "scalar":
            evals = (pseq(samples).map(lambda x: f(x)).to_list())
        else:
            evals = f(samples)
        return samples, evals
    #print("  Calculating lf and hf on N_low")
    [_, evals] = calc_evals(lf, N_low)
    #print("  Calculating f_bar")
    [ftild_bar, _] = monte_carlo(evals)

    # 2. Calculate lf and hf on N_high
    #print("  Calculating lf and hf on N_high")
    [_, hf_evals] = calc_evals(hf, N_high)
    [hf_exp, hf_var] = monte_carlo(hf_evals)
    [_, lf_evals] = calc_evals(lf, N_high)
    [lf_exp, lf_var] = monte_carlo(lf_evals)

    # 3. Calc Covariance
    #print("  Calculating covariance")
    cov = (1/N_high) * (pseq(range(N_high))
                        .map(lambda i : (hf_evals[i] - hf_exp) * (lf_evals[i] - lf_exp))
                        .sum())
    # 4. Calc sigma_hat
    #print("  Calculating sigma_hat")
    sigma_hat = (1/N_high) * (pseq(range(N_high))
                              .map(lambda i: (lf_evals[i] - lf_exp)**2)
                              .sum())

    # 5. Calc Alpha
    #print("  Calculating alpha")
    alpha = cov/sigma_hat

    # 6. Calc Gbar
    #print("  Calculating gbar")
    gbar = ((1/N_high)
            * (pseq(range(N_high))
               .map(lambda i: hf_evals[i] - alpha*lf_evals[i])
               .sum())
            + (alpha * ftild_bar))

    # 7. Calc Gvar
    #print("  Calculating gvar")
    gvar = hf_var - 2*alpha*cov + alpha**2 * lf_var
    return [gbar, gvar, alpha]

# def mult-level(fs=[f0,f1,...], Ns=[N0,N1,...], C=[C0,C1,...]):
#     1. Step 1: For all submodels (f1,..fS), calculate expected values using (N1,...,NS)
#     fs_bar = map(lambda fi,Ni : calc_evals(fi,Ni), fs, Ns)
#     2. Step 2: Optimize Nf_tild
#        [_,var0] = fs_bar[0]
#        [_,var1] = fs_bar[1]
#        V[f0 - f1] = (v[f0] - V[f1]/N1) * N0
#        Nf1 = sqrt(Var(f1)/C1) / sqrt(Var(f0-f1)/C0) * NF
#     3. Make recurrence relation, and build N_opt = [N0, N1_opt, ...]
#     4. Build expectation
#     5. Build variance

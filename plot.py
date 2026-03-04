import os
import ast
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from functional import seq

CWD = os.path.dirname(os.path.abspath(__file__))

def read_dictionary_file(filename):
    with open(os.path.join(CWD,filename), "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = ast.literal_eval(line)
        data.append(record)

    unique_N = sorted({d["N"] for d in data})
    grouped_dicts = group(data)
    return unique_N, grouped_dicts

def group(flat_data):
    unique_N = sorted({d["N"] for d in flat_data})
    # Group by N
    grouped_dict = defaultdict(list)
    for d in flat_data:
        grouped_dict[d["N"]].append(d)
    return [grouped_dict[N] for N in unique_N]

def get_all(d,key):
    return list(map(lambda met: met[key], d))

def calc_fbar(run):
    exps = get_all(run, 'exp')
    return np.sqrt(np.var(exps, ddof=1))

def ref_line(Ns):
    return list(map(lambda n : 0 if n == 0 else n ** -0.5, Ns))

def plot_rmse(title, Ns, *rmses):
    plt.figure()
    plt.plot(Ns, ref_line(Ns), label="Reference:")
    for rmse in rmses:
        plt.plot(Ns, rmse['rmse'], label=rmse['label'])
    plt.xlabel("N")
    plt.ylabel("RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(title)



if __name__ == "__main__":
    # MC
    [mc_N, mc_runs] = read_dictionary_file("out/mc_b3.dict")
    mc_var = list(map(calc_fbar, mc_runs))

    # CV
    [cv_N, cv_runs] = read_dictionary_file("out/cv_b3_alpha_p10.dict")
    #alphas = (seq(cv_runs)
    #          .map(lambda run: get_all(run, 'alpha'))
    #          .flatten()
    #          .to_list())
    #print(np.mean(alphas))
    cv_var = list(map(calc_fbar, cv_runs))

    # MLMC
    #[mlmc_N, mlmc_runs] = read_dictionary_file("out/mlmc_b3_higher_k.dict")
    #mlmc_var = list(map(calc_fbar, mlmc_runs))

    plot_rmse("RMSE Analytical Beta=0.3: alpha* + 1.0", mc_N,
              {'rmse' : mc_var, 'label':"Monte-Carlo"},
              {'rmse' : cv_var, 'label':"Control Variate"}
              #{'rmse' : mlmc_var, 'label' : "Multi-Level Monte-Carlo"}
              )
    plt.show()

### The code to get the meaningful results from EPO model
import numpy as np
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chapter_1.epo_model import EPO_Params, Sim_results, gen_sim

### Functions for results presentation

def final_pi(sim: Sim_results):
    """Extract final network PI from results"""
    return sim.history[-1]['pi']

def converg_steps(sims):
    """Extract convergence steps from the results"""
    return[sim.convergence_step for sim in sims]

def aggreg_res(simulations: List[Sim_results], params: EPO_Params):
    """Get meaningful statistics from the results"""

    pi_fs = [final_pi(sim) for sim in simulations]
    conv = [sim.convergence_step for sim in simulations]

    avg_pi = np.mean(pi_fs)
    std_pi = np.std(pi_fs)
    avg_conv = np.mean(conv)

    #parameter summaries
    lam_mean = np.mean(params.lambda_range)
    phi_mean = np.mean(params.phi_range)
    aii_mean = np.mean(params.a_ii_range)

    return {'lambda_mean': lam_mean,
            'phi_mean': phi_mean,
            'aii_mean': aii_mean,
            'p': params.p,
            'avg_pi': avg_pi,
            'std_pi': std_pi,
            'avg_conv': avg_conv
            }


def avg_pi_time(sims: List[Sim_results]):
    """Collect the PI over time for plotting """
    max_len = max(len(sim.history) for sim in sims)

    pi_matrix = []

    for sim in sims:
        pi_series = [h['pi'] for h in sim.history]
        if len(pi_series) < max_len:
            pi_series += [pi_series[-1]]*(max_len-len(pi_series))
        pi_matrix.append(pi_series)

    return np.mean(pi_matrix, axis=0)

### Batch simulations functions

def batch_sim(n_sims = 50):
    """Run batch simulations with parameter sweeps"""
    all_results = []
    all_pi_curves = []

    for lam in lambda_sets:
        for phi in phi_sets:
            for p in p_sets:
                for aii in aii_sets:

                    params = EPO_Params(lambda_range=lam,
                                        phi_range=phi,
                                        p=p, a_ii_range=aii)
                    sims = gen_sim(params, n_simulations=n_sims)
                    agg = aggreg_res(sims, params)
                    all_results.append(agg)

                    pi_curve = avg_pi_time(sims)
                    all_pi_curves.append({'params': agg,
                                          'pi_curve': pi_curve})
                    print(f"Done: λ={lam}, φ={phi}, p={p}, aii={aii}")

    return all_results, all_pi_curves

### for parameter sweeps
lambda_sets = [(0.1,0.2), (0.4,0.6), (0.8,1.0)]
phi_sets = [(0.1,0.2), (0.4,0.6), (0.8,1.0)]
p_sets = [0.2, 0.5, 0.8]
aii_sets = [(0.1,0.2), (0.4,0.6), (0.8,1.0)]


if __name__ == "__main__":
    results, pi_curves = batch_sim(n_sims=50)

    #print("\nAGGREGATED RESULTS:")
    #for r in results:
    #    print(r)
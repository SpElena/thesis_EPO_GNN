### This code gets plots & tables to be used in analysis and thesis presentation

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_1.epo_exp import batch_sim

### Get plots with some fixed parameters
def plot_pi_by_par(pi_curves, fixed_pars=None, save_path=None, show=True):
    """Fix some parameter(s) and plot"""

    if fixed_pars is None:
        fixed_pars = {}

    plt.figure()

    for entry in pi_curves:
        params = entry['params']

        # Match with tolerance
        match = True
        for key, value in fixed_pars.items():
            if not np.isclose(params[key], value, atol=1e-3):
                match = False
                break

        if not match:
            continue

        curve = entry['pi_curve']

        # Build label dynamically
        label_parts = []
        if 'lambda_mean' not in fixed_pars:
            label_parts.append(f"λ={params['lambda_mean']:.2f}")
        if 'phi_mean' not in fixed_pars:
            label_parts.append(f"φ={params['phi_mean']:.2f}")
        if 'p' not in fixed_pars:
            label_parts.append(f"p={params['p']}")
        if 'aii_mean' not in fixed_pars:
            label_parts.append(f"a_ii={params['aii_mean']:.2f}")

        label = ", ".join(label_parts)

        curve = np.array(curve) + np.random.normal(0, 0.005, len(curve)) #to see the overlapping
        plt.plot(curve, label=label, linestyle='--', marker = 'o', markersize=3)

    # Labels outside loop
    plt.xlabel("Time step")
    plt.ylabel("Average PI")

    title_params = ", ".join([f"{k}={v}" for k, v in fixed_pars.items()])
    plt.title(f"PI over time ({title_params})")

    plt.legend(fontsize=8)
    plt.grid()

    # Save
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence(results, fixed_pars=None, save_path=None, show=True):
    """Plot convergence vs varying parameter"""

    if fixed_pars is None:
        fixed_pars = {}

    # Filter results
    filtered = []
    for r in results:
        match = True
        for key, value in fixed_pars.items():
            if not np.isclose(r[key], value):
                match = False
                break
        if match:
            filtered.append(r)

    if len(filtered) == 0:
        print("No matching results found.")
        return

    # Identify varying parameter
    keys = ['lambda_mean', 'phi_mean', 'aii_mean','p']
    varying_key = [k for k in keys if k not in fixed_pars]

    if len(varying_key) != 1:
        print("Please fix exactly three parameters.")
        return

    varying_key = varying_key[0]

    # Sort by varying parameter
    filtered = sorted(filtered, key=lambda x: x[varying_key])

    x = [r[varying_key] for r in filtered]
    y = [r['avg_conv'] for r in filtered]

    plt.figure()
    plt.plot(x, y, marker='o')

    # Labels
    label_map = {
        'lambda_mean': 'λ',
        'phi_mean': 'φ',
        'p': 'p',
        'aii_mean': r'$a_{ii}$'
    }

    plt.xlabel(label_map[varying_key])
    plt.ylabel("Average Convergence Step")

    title_params = ", ".join([f"{k}={v}" for k, v in fixed_pars.items()])
    plt.title(f"Convergence vs {label_map[varying_key]} ({title_params})")

    plt.grid()

    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


### Generate a table with results for each plot
def filtered_table(results, fixed_pars, varying_key):
    """Filter results and return sorted table data"""

    filtered = []
    for r in results:
        match = True
        for key, value in fixed_pars.items():
            if not np.isclose(r[key], value):
                match = False
                break
        if match:
            filtered.append(r)

    # sort by varying parameter
    filtered = sorted(filtered, key=lambda x: x[varying_key])

    return filtered


### LateX table for one experiment

def table_to_latex(data, varying_key, caption):
    """Convert filtered data into LaTeX table"""

    label_map = {
        'lambda_mean': r'$\lambda$',
        'phi_mean': r'$\phi$',
        'aii_mean': r'$a_{ii}$',
        'p': r'$p$'
    }

    header = f"""
\\begin{{table}}[H]
\\centering
\\small
\\begin{{tabular}}{{cccc}}
\\hline
{label_map[varying_key]} & Avg PI & Std PI & Avg Conv \\\\
\\hline
"""

    rows = ""
    for r in data:
        rows += f"{r[varying_key]:.2f} & {r['avg_pi']:.3f} & {r['std_pi']:.3f} & {r['avg_conv']:.2f} \\\\\n"

    footer = f"""\\hline
\\end{{tabular}}
\\caption{{{caption}}}
\\end{{table}}
"""

    return header + rows + footer






### Execution

if __name__ == "__main__":
    # Run simulations
    results, pi_curves = batch_sim(n_sims=50)

    ### Plotting PI

    # Fix all but lambda
    plot_pi_by_par(pi_curves,
    fixed_pars={'phi_mean': 0.5, 'p': 0.5, 'aii_mean': 0.5},
    save_path="figures/pi_effect_lambda.png",
    show=False)
    
    # Fix all but phi
    plot_pi_by_par(
    pi_curves,
    fixed_pars={'lambda_mean': 0.5, 'p': 0.5, 'aii_mean': 0.5},
    save_path="figures/pi_effect_phi.png",
    show=False)
    
    # Fix all but p
    plot_pi_by_par(
    pi_curves,
    fixed_pars={'lambda_mean': 0.5, 'phi_mean': 0.5, 'aii_mean': 0.5},
    save_path="figures/pi_effect_p.png",
    show=False)

    # Fix all but aii
    plot_pi_by_par(
    pi_curves,
    fixed_pars={'lambda_mean': 0.5, 'phi_mean': 0.5, 'p': 0.5},
    save_path="figures/pi_effect_aii.png",
    show=False)
    
    ### Plotting Convergence

    # vs lambda
    plot_convergence(results,
    fixed_pars={'phi_mean': 0.5, 'p': 0.5, 'aii_mean': 0.5},
    save_path="figures/conv_effect_lambda.png",
    show=False)
    
    # vs phi
    plot_convergence(results,
    fixed_pars={'lambda_mean': 0.5, 'p': 0.5, 'aii_mean': 0.5},
    save_path="figures/conv_effect_phi.png",
    show=False)
    
    # vs p
    plot_convergence(results,
    fixed_pars={'lambda_mean': 0.5, 'phi_mean': 0.5, 'aii_mean': 0.5},
    save_path="figures/conv_effect_p.png",
    show=False)

    # vs aii
    plot_convergence(results,
    fixed_pars={'lambda_mean': 0.5, 'phi_mean': 0.5, 'p': 0.5},
    save_path="figures/conv_effect_aii.png",
    show=False)

    # Generate LaTeX tables
    lambda_data = filtered_table(results,
        fixed_pars={'phi_mean': 0.5, 'p': 0.5, 'aii_mean': 0.5},
        varying_key='lambda_mean')

    latex_lambda = table_to_latex(lambda_data,
        varying_key='lambda_mean',
        caption='Effect of $\\lambda$ on PI and convergence ($\phi=0.5$, $p=0.5$, $a_{ii}=0.5$)')

    phi_data = filtered_table(results,
        fixed_pars={'lambda_mean': 0.5, 'p': 0.5, 'aii_mean': 0.5},
        varying_key='phi_mean')

    latex_phi = table_to_latex(phi_data,
        varying_key='phi_mean',
        caption='Effect of $\\phi$ on PI and convergence ($\lambda=0.5$, $p=0.5$, $a_{ii}=0.5$)')
    
    p_data = filtered_table(results,
        fixed_pars={'lambda_mean': 0.5, 'phi_mean': 0.5, 'aii_mean': 0.5},
        varying_key='p')

    latex_p = table_to_latex(p_data,
        varying_key='p',
        caption='Effect of network density $p$ on PI and convergence ($\lambda=0.5$, $\phi=0.5$, $a_{ii}=0.5$)')
    
    aii_data = filtered_table(results,
        fixed_pars={'lambda_mean': 0.5, 'phi_mean': 0.5, 'p': 0.5},
        varying_key='aii_mean')

    latex_aii = table_to_latex(aii_data,
        varying_key='aii_mean',
        caption='Effect of $a_{ii}$ on PI and convergence ($\\lambda=0.5$, $\\phi=0.5$, $p=0.5$)')

    with open("tables/tables_lambda.tex", "w") as f:
        f.write(latex_lambda)

    with open("tables/tables_phi.tex", "w") as f:
        f.write(latex_phi)

    with open("tables/tables_p.tex", "w") as f:
        f.write(latex_p)

    with open("tables/tables_aii.tex", "w") as f:
        f.write(latex_aii)
    

    
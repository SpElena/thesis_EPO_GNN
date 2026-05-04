"""
Evaluation script for GNN-EPO model

Part 1: Exact mode validation — here we confirm that GNN-EPO is a true surrogate of original EPO
Part 2: Estimator diagnostics — we observe what the model learned about the estimated parameteres 
(spoiler, it actually reveals that all estimators collapse to constants :(
Part 3: Missing parameter evaluation — here we measure epo_gnn's robustness to missing parameters
(i.e. how each affect the PI and convergence)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chapter_1.epo_model import EPO_Params, create_initial_graph, net_pi, Sim_results
from chapter_2.gnn_epo_model import EPO_GNN
from chapter_2.gnn_epo_train import graph_to_data

# Output directory (for all results from this file)
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

def save_path(filename):
    return os.path.join(results_dir, filename)


# =============================
# GNN-EPO rollout (step-by-step with forward() for missing params)

def rollout_gnn(model, G, params, mask, max_steps=20):
    """
    Run a full GNN-EPO simulation step-by-step using forward() 
    This is used for missing-parameter evaluation since forward() activates the learned estimators
    when parameters are masked (missing)"""

    nodes = sorted(G.nodes())
    private = np.array([G.nodes[n]['private_op'] for n in nodes])
    expressed = np.array([G.nodes[n]['expressed_op'] for n in nodes])
    
    history = []
    
    for step in range(max_steps):
        for i, n in enumerate(nodes):
            G.nodes[n]['private_op'] = private[i]
            G.nodes[n]['expressed_op'] = expressed[i]
        pi = net_pi(G)
        
        history.append({
            'step': step,
            'private': private.copy().tolist(),
            'expressed': expressed.copy().tolist(),
            'pi': pi
        })
        
        data = graph_to_data(G, private, expressed)
        with torch.no_grad():
            x_next, y_next = model(data, mask=mask)
        
        private_new = np.where(x_next.numpy() > 0, 1, -1).flatten()
        expressed_new = np.where(y_next.numpy() > 0, 1, -1).flatten()
        
        if np.array_equal(private, private_new) and np.array_equal(expressed, expressed_new):
            break
        
        private = private_new
        expressed = expressed_new
    
    for i, n in enumerate(nodes):
        G.nodes[n]['private_op'] = private[i]
        G.nodes[n]['expressed_op'] = expressed[i]
    history.append({'step': len(history), 'pi': net_pi(G)})
    
    return Sim_results(
        graph=G, history=history,
        convergence_step=len(history) - 1,
        private_final=private.tolist(),
        expressed_final=expressed.tolist()
    )


# =============================
# Original EPO run on specific graph
def run_sim_with_graph(G, params):
    """Run original EPO a specific graph (for comparison to epo-gnn)"""
    from chapter_1.epo_model import private_upd, expressed_upd
    
    history = []
    
    for step in range(params.steps):
        nodes = list(G.nodes())
        changed = False
        updates = {}
        
        for node in nodes:
            new_private = private_upd(G, node, params)
            new_expressed = expressed_upd(G, node, new_private, params)
            
            if new_private != G.nodes[node]['private_op'] or new_expressed != G.nodes[node]['expressed_op']:
                changed = True
            updates[node] = (new_private, new_expressed)
        
        for node, (priv, expr) in updates.items():
            G.nodes[node]['private_op'] = priv
            G.nodes[node]['expressed_op'] = expr
        
        pi = net_pi(G)
        history.append({
            'step': step,
            'private': [G.nodes[n]['private_op'] for n in sorted(G.nodes())],
            'expressed': [G.nodes[n]['expressed_op'] for n in sorted(G.nodes())],
            'pi': pi
        })
        
        if not changed:
            break
    
    nodes_sorted = sorted(G.nodes())
    return Sim_results(
        graph=G, history=history,
        convergence_step=len(history) - 1,
        private_final=[G.nodes[n]['private_op'] for n in nodes_sorted],
        expressed_final=[G.nodes[n]['expressed_op'] for n in nodes_sorted]
    )


# =============================
# Average PI over time
def avg_pi_time(sims):
    """Average PI over time across multiple simulations"""
    max_len = max(len(sim.history) for sim in sims)
    pi_matrix = []
    for sim in sims:
        pi_series = [h['pi'] for h in sim.history]
        if len(pi_series) < max_len:
            pi_series += [pi_series[-1]] * (max_len - len(pi_series))
        pi_matrix.append(pi_series)
    return np.mean(pi_matrix, axis=0)


# =============================
# LaTeX table generators
def generate_exact_validation_table(orig_pi, gnn_pi, orig_conv, gnn_conv, all_match, n_sims):
    """Generate LaTeX table for Part 1: Exact Mode Validation"""
    latex = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Exact Mode Validation: Original EPO vs GNN-EPO (All Parameters Provided)}}
\\label{{tab:exact_validation}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Original EPO}} & \\textbf{{GNN-EPO}} \\\\
\\hline
Final PI & {orig_pi:.4f} & {gnn_pi:.4f} \\\\
Convergence Steps & {orig_conv:.2f} & {gnn_conv:.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(save_path('table_exact_validation.tex'), 'w') as f:
        f.write(latex)
    print(f"Saved table: {save_path('table_exact_validation.tex')}")
    return latex


def generate_estimator_diagnostic_table(all_true_lam, all_est_lam, all_true_phi, 
                                         all_est_phi, all_true_aii, all_est_aii):
    """Generate LaTeX table for Part 2: Estimator Diagnostics"""
    corr_lam = np.corrcoef(all_true_lam, all_est_lam)[0, 1]
    corr_phi = np.corrcoef(all_true_phi, all_est_phi)[0, 1]
    corr_aii = np.corrcoef(all_true_aii, all_est_aii)[0, 1]
    
    latex = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Estimator Diagnostics: True vs Estimated Parameter Values}}
\\label{{tab:estimator_diagnostics}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{Parameter}} & \\textbf{{True Mean}} & \\textbf{{Est Mean}} & \\textbf{{True Std}} & \\textbf{{Est Std}} & \\textbf{{Correlation}} \\\\
\\hline
$\\lambda$ (Susceptibility) & {all_true_lam.mean():.4f} & {all_est_lam.mean():.4f} & {all_true_lam.std():.4f} & {all_est_lam.std():.4f} & {corr_lam:.4f} \\\\
$\\phi$ (Resilience) & {all_true_phi.mean():.4f} & {all_est_phi.mean():.4f} & {all_true_phi.std():.4f} & {all_est_phi.std():.4f} & {corr_phi:.4f} \\\\
$a_{{ii}}$ (Self-confidence) & {all_true_aii.mean():.4f} & {all_est_aii.mean():.4f} & {all_true_aii.std():.4f} & {all_est_aii.std():.4f} & {corr_aii:.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(save_path('table_estimator_diagnostics.tex'), 'w') as f:
        f.write(latex)
    print(f"Saved table: {save_path('table_estimator_diagnostics.tex')}")
    return latex


def generate_missing_params_table(results):
    """Generate LaTeX table for Part 3: Missing Parameter Scenarios"""
    
    # Map to LaTeX-safe names
    latex_names = {
        'Exact (all params)': 'Exact (all params)',
        'Missing λ': 'Missing $\\lambda$',
        'Missing φ': 'Missing $\\phi$',
        'Missing a_ii': 'Missing $a_{ii}$',
        'Missing λ, φ': 'Missing $\\lambda$, $\\phi$',
        'Missing λ, a_ii': 'Missing $\\lambda$, $a_{ii}$',
        'Missing φ, a_ii': 'Missing $\\phi$, $a_{ii}$',
        'Missing all': 'Missing all',
    }
    
    latex = """
\\begin{table}[H]
\\centering
\\caption{Pluralistic Ignorance and Convergence Across Missing-Parameter Scenarios}
\\label{tab:missing_params}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Scenario} & \\textbf{Avg PI} & \\textbf{Std PI} & \\textbf{Avg Conv} & \\textbf{Std Conv} \\\\
\\hline
"""
    for key in ['Exact (all params)', 'Missing λ', 'Missing φ', 'Missing a_ii',
                 'Missing λ, φ', 'Missing λ, a_ii', 'Missing φ, a_ii', 'Missing all']:
        if key in results:
            data = results[key]
            name = latex_names.get(key, key)
            latex += f"{name} & {data['avg_pi']:.4f} & {data['std_pi']:.4f} & {data['avg_conv']:.2f} & {data['std_conv']:.2f} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    with open(save_path('table_missing_params.tex'), 'w') as f:
        f.write(latex)
    print(f"Saved table: {save_path('table_missing_params.tex')}")
    return latex


def generate_parameter_importance_table(results):
    """Generate LaTeX table for parameter importance analysis"""
    baseline = results.get('Exact (all params)')
    if baseline is None:
        return ""
    
    baseline_pi = baseline['avg_pi']
    
    latex_names = {
        'Missing λ': 'Missing $\\lambda$',
        'Missing φ': 'Missing $\\phi$',
        'Missing a_ii': 'Missing $a_{ii}$',
        'Missing λ, φ': 'Missing $\\lambda$, $\\phi$',
        'Missing λ, a_ii': 'Missing $\\lambda$, $a_{ii}$',
        'Missing φ, a_ii': 'Missing $\\phi$, $a_{ii}$',
        'Missing all': 'Missing all',
    }
    
    latex = """
\\begin{table}[H]
\\centering
\\caption{Parameter Importance: PI Error Relative to Exact Mode Baseline}
\\label{tab:parameter_importance}
\\begin{tabular}{lcc}
\\hline
\\textbf{Scenario} & \\textbf{Absolute Error} & \\textbf{\\% Change} \\\\
\\hline
"""
    for key in ['Missing λ', 'Missing φ', 'Missing a_ii',
                 'Missing λ, φ', 'Missing λ, a_ii', 'Missing φ, a_ii', 'Missing all']:
        if key in results and key != 'Exact (all params)':
            error = abs(results[key]['avg_pi'] - baseline_pi)
            pct = (error / baseline_pi) * 100 if baseline_pi > 0 else 0
            name = latex_names.get(key, key)
            latex += f"{name} & {error:.4f} & {pct:.1f}\\% \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    with open(save_path('table_parameter_importance.tex'), 'w') as f:
        f.write(latex)
    print(f"Saved table: {save_path('table_parameter_importance.tex')}")
    return latex


# ======================
# PART 1: Exact Mode validation against original EPO
def validate_exact_mode(model, params, n_sims=30):
    """Validate that GNN-EPO in exact mode produces identical results to original EPO. 
    Uses simulate() for exact matching"""

    print("\n" + "=" * 70)
    print("PART 1: Exact mode validation")
    print("  Original EPO vs GNN-EPO (all parameters provided)")
    print("=" * 70)
    
    orig_sims = []
    gnn_sims = []
    
    for seed in range(n_sims):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        params_copy = EPO_Params(
            n=params.n, p=params.p,
            lambda_range=params.lambda_range,
            phi_range=params.phi_range,
            a_ii_range=params.a_ii_range,
            steps=params.steps, seed=seed
        )
        
        G = create_initial_graph(params_copy)
        nodes = sorted(G.nodes())
        
        G_orig = copy.deepcopy(G)
        sim_orig = run_sim_with_graph(G_orig, params_copy)
        orig_sims.append(sim_orig)
        
        G_gnn = copy.deepcopy(G)
        private = np.array([G_gnn.nodes[n]['private_op'] for n in nodes])
        expressed = np.array([G_gnn.nodes[n]['expressed_op'] for n in nodes])
        data = graph_to_data(G_gnn, private, expressed)
        
        with torch.no_grad():
            final_x, history_tensors = model.simulate(data, steps=params.steps, mask=None)
        
        history = []
        for step, x in enumerate(history_tensors):
            for i, n in enumerate(nodes):
                G_gnn.nodes[n]['private_op'] = int(x[i, 0].item())
                G_gnn.nodes[n]['expressed_op'] = int(x[i, 1].item())
            pi = net_pi(G_gnn)
            history.append({
                'step': step,
                'private': x[:, 0].tolist(),
                'expressed': x[:, 1].tolist(),
                'pi': pi
            })
        
        sim_gnn = Sim_results(
            graph=G_gnn, history=history,
            convergence_step=len(history) - 1,
            private_final=final_x[:, 0].tolist(),
            expressed_final=final_x[:, 1].tolist()
        )
        gnn_sims.append(sim_gnn)
    
    orig_pi = np.mean([s.history[-1]['pi'] for s in orig_sims])
    gnn_pi = np.mean([s.history[-1]['pi'] for s in gnn_sims])
    orig_conv = np.mean([s.convergence_step for s in orig_sims])
    gnn_conv = np.mean([s.convergence_step for s in gnn_sims])
    
    all_match = all(
        abs(s_o.history[-1]['pi'] - s_g.history[-1]['pi']) < 1e-6 and
        s_o.convergence_step == s_g.convergence_step
        for s_o, s_g in zip(orig_sims, gnn_sims)
    )
    
    print(f"\n{'':<25} {'Original EPO':<15} {'GNN-EPO':<15} {'Match':<10}")
    print("-" * 65)
    print(f"{'Avg Final PI':<25} {orig_pi:<15.4f} {gnn_pi:<15.4f} {'✓' if abs(orig_pi-gnn_pi)<1e-6 else '✗'}")
    print(f"{'Avg Conv Steps':<25} {orig_conv:<15.2f} {gnn_conv:<15.2f} {'✓' if abs(orig_conv-gnn_conv)<1e-6 else '✗'}")
    print(f"{'All seeds match':<25} {'':<15} {'':<15} {'✓' if all_match else '✗'}")
    
    if all_match:
        print("\n GNN-EPO is a perfect surrogate of original EPO")
    
    # Generate LaTeX table
    generate_exact_validation_table(orig_pi, gnn_pi, orig_conv, gnn_conv, all_match, n_sims)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(avg_pi_time(orig_sims), 'b-o', label='Original EPO', linewidth=2, markersize=4)
    plt.plot(avg_pi_time(gnn_sims), 'r--s', label='GNN-EPO (exact)', linewidth=2, markersize=4)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Proportion of Nodes with PI", fontsize=12)
    plt.title(f"Validation: GNN-EPO = Original EPO ({n_sims} seeds)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path('fig_exact_validation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path('fig_exact_validation.png')}")
    
    return all_match


# =============================
# PART 2: Estimator Diagnostics
def diagnose_all_estimators(model, params, n_runs=30):
    """Compare true vs estimated parameter values"""

    print("\n" + "=" * 70)
    print("PART 2: Estimator diagnostics")
    print("  Comparing true vs estimated parameter values")
    print("=" * 70)
    
    all_true_lam, all_est_lam = [], []
    all_true_phi, all_est_phi = [], []
    all_true_aii, all_est_aii = [], []
    
    for seed in range(n_runs):
        params_copy = EPO_Params(
            n=params.n, p=params.p,
            lambda_range=params.lambda_range,
            phi_range=params.phi_range,
            a_ii_range=params.a_ii_range,
            steps=1, seed=seed
        )
        
        G = create_initial_graph(params_copy)
        nodes = sorted(G.nodes())
        private = np.array([G.nodes[n]['private_op'] for n in nodes])
        expressed = np.array([G.nodes[n]['expressed_op'] for n in nodes])
        
        all_true_lam.extend([G.nodes[n]['lambda'] for n in nodes])
        all_true_phi.extend([G.nodes[n]['phi'] for n in nodes])
        all_true_aii.extend([G.nodes[n]['a_ii'] for n in nodes])
        
        data = graph_to_data(G, private, expressed)
        model.train()
        with torch.no_grad():
            context = model._compute_context(data.x, data)
            all_est_lam.extend(model.lambda_estimator(context).numpy())
            all_est_phi.extend(model.phi_estimator(context).numpy())
            all_est_aii.extend(model.aii_estimator(context).numpy())
    
    all_true_lam = np.array(all_true_lam)
    all_est_lam = np.array(all_est_lam)
    all_true_phi = np.array(all_true_phi)
    all_est_phi = np.array(all_est_phi)
    all_true_aii = np.array(all_true_aii)
    all_est_aii = np.array(all_est_aii)
    
    corr_lam = np.corrcoef(all_true_lam, all_est_lam)[0, 1]
    corr_phi = np.corrcoef(all_true_phi, all_est_phi)[0, 1]
    corr_aii = np.corrcoef(all_true_aii, all_est_aii)[0, 1]
    
    print(f"\n{'Parameter':<12} {'True Mean':<12} {'Est Mean':<12} {'True Std':<12} {'Est Std':<12} {'Correlation':<12}")
    print("-" * 72)
    print(f"{'λ':<12} {all_true_lam.mean():<12.4f} {all_est_lam.mean():<12.4f} {all_true_lam.std():<12.4f} {all_est_lam.std():<12.4f} {corr_lam:<12.4f}")
    print(f"{'φ':<12} {all_true_phi.mean():<12.4f} {all_est_phi.mean():<12.4f} {all_true_phi.std():<12.4f} {all_est_phi.std():<12.4f} {corr_phi:<12.4f}")
    print(f"{'a_ii':<12} {all_true_aii.mean():<12.4f} {all_est_aii.mean():<12.4f} {all_true_aii.std():<12.4f} {all_est_aii.std():<12.4f} {corr_aii:<12.4f}")
    
    print("\n  All three estimators collapse to near-constant values:")
    print(f"    λ → {all_est_lam.mean():.4f} (true: {all_true_lam.mean():.4f})")
    print(f"    φ → {all_est_phi.mean():.4f} (true: {all_true_phi.mean():.4f})")
    print(f"    a_ii → {all_est_aii.mean():.4f} (true: {all_true_aii.mean():.4f})")
    
    # Generate LaTeX table
    generate_estimator_diagnostic_table(all_true_lam, all_est_lam, all_true_phi,
                                         all_est_phi, all_true_aii, all_est_aii)
    
    # Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, true, est, corr, name in zip(
        axes,
        [all_true_lam, all_true_phi, all_true_aii],
        [all_est_lam, all_est_phi, all_est_aii],
        [corr_lam, corr_phi, corr_aii],
        ['λ (Susceptibility)', 'φ (Resilience)', 'a_ii (Self-confidence)']
    ):
        ax.scatter(true, est, alpha=0.3, s=10, color='#3498db', edgecolors='none')
        ax.plot([0.4, 0.6], [0.4, 0.6], 'r--', linewidth=2, label='Perfect estimation')
        ax.axhline(y=est.mean(), color='red', linestyle=':', alpha=0.7, 
                   label=f'Est mean ({est.mean():.3f})')
        ax.set_xlabel('True Value', fontsize=11)
        ax.set_ylabel('Estimated Value', fontsize=11)
        ax.set_title(f'{name}\n(r = {corr:.3f})', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.38, 0.62)
    plt.tight_layout()
    plt.savefig(save_path('fig_all_estimators.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path('fig_all_estimators.png')}")
    
    return (all_true_lam, all_est_lam), (all_true_phi, all_est_phi), (all_true_aii, all_est_aii)


# =============================
# PART 3: Missing parameters' evaluation
def run_evaluation(model, params, n_sims=30):
    """Evaluate GNN-EPO robustness to missing parameters"""

    print("\n" + "=" * 70)
    print("PART 3: MISSING PARAMETER EVALUATION")
    print("  Measuring robustness of learned dynamics to missing parameters")
    print("=" * 70)
    
    scenarios = {
        'Exact (all params)':     None,
        'Missing λ':              {'lambda': False, 'phi': True, 'aii': True},
        'Missing φ':              {'lambda': True, 'phi': False, 'aii': True},
        'Missing a_ii':           {'lambda': True, 'phi': True, 'aii': False},
        'Missing λ, φ':           {'lambda': False, 'phi': False, 'aii': True},
        'Missing λ, a_ii':        {'lambda': False, 'phi': True, 'aii': False},
        'Missing φ, a_ii':        {'lambda': True, 'phi': False, 'aii': False},
        'Missing all':            {'lambda': False, 'phi': False, 'aii': False},
    }
    
    results = {}
    
    for name, mask in scenarios.items():
        print(f"Running: {name}")
        sims = []
        
        for seed in range(n_sims):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            params_copy = EPO_Params(
                n=params.n, p=params.p,
                lambda_range=params.lambda_range,
                phi_range=params.phi_range,
                a_ii_range=params.a_ii_range,
                steps=params.steps, seed=seed
            )
            
            G = create_initial_graph(params_copy)
            G_gnn = copy.deepcopy(G)
            sim = rollout_gnn(model, G_gnn, params_copy, mask)
            sims.append(sim)
        
        final_pis = [s.history[-1]['pi'] for s in sims]
        conv_steps = [s.convergence_step for s in sims]
        
        results[name] = {
            'avg_pi': np.mean(final_pis),
            'std_pi': np.std(final_pis),
            'avg_conv': np.mean(conv_steps),
            'std_conv': np.std(conv_steps),
            'pi_curve': avg_pi_time(sims),
        }
        
        print(f"  Avg PI: {results[name]['avg_pi']:.4f}, Avg Conv: {results[name]['avg_conv']:.2f}")
    
    return results


# =============================
# Print tables
def print_summary_table(results):
    """Print a summary table of all scenarios."""
    print("\n" + "=" * 80)
    print("SUMMARY: MISSING PARAMETER SCENARIOS")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Avg PI':<10} {'Std PI':<10} {'Avg Conv':<10} {'Std Conv':<10}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<25} {data['avg_pi']:<10.4f} {data['std_pi']:<10.4f} "
              f"{data['avg_conv']:<10.2f} {data['std_conv']:<10.2f}")
    print("=" * 80)


def print_parameter_importance(results):
    """Print PI error relative to exact mode baseline."""
    baseline = results.get('Exact (all params)')
    if baseline is None:
        return
    
    baseline_pi = baseline['avg_pi']
    
    print("\nPARAMETER IMPORTANCE (Error vs Exact Mode Baseline):")
    print("-" * 55)
    print(f"{'Scenario':<25} {'Abs Error':<12} {'% Change':<10}")
    print("-" * 55)
    for name, data in results.items():
        if name == 'Exact (all params)':
            continue
        error = abs(data['avg_pi'] - baseline_pi)
        pct = (error / baseline_pi) * 100 if baseline_pi > 0 else 0
        print(f"{name:<25} {error:<12.4f} {pct:<10.1f}%")
    print("-" * 55)


# =============================
# Plots
def plot_pi_curves(results):
    """Plot PI curves for all scenarios."""
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, data), color in zip(results.items(), colors):
        plt.plot(data['pi_curve'], label=name, color=color, linewidth=2, marker='o', markersize=3)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Proportion of Nodes with PI", fontsize=12)
    plt.title("PI Curves Across Missing-Parameter Scenarios", fontsize=14)
    plt.legend(fontsize=8, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path('fig_pi_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path('fig_pi_curves.png')}")


def plot_parameter_importance(results):
    """Bar chart: PI error when each parameter is missing."""
    baseline = results['Exact (all params)']['avg_pi']
    scenarios = ['Missing λ', 'Missing φ', 'Missing a_ii', 
                 'Missing λ, φ', 'Missing λ, a_ii', 'Missing φ, a_ii', 'Missing all']
    errors = [abs(results[s]['avg_pi'] - baseline) if s in results else 0 for s in scenarios]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(scenarios, errors, color=['#3498db', '#2ecc71', '#e74c3c', 
                                              '#f39c12', '#9b59b6', '#1abc9c', '#95a5a6'])
    plt.ylabel("Absolute Error in Final PI", fontsize=12)
    plt.title("Impact of Missing Parameters on PI Prediction", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    for bar, err in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{err:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path('fig_parameter_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path('fig_parameter_importance.png')}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Loading trained model...")
    model = EPO_GNN(hidden_dim=32)
    model.load_state_dict(torch.load('trained_EPO_GNN.pth', map_location='cpu'))
    print("Model loaded.\n")
    
    params = EPO_Params(
        n=30, p=0.5,
        lambda_range=(0.4, 0.6),
        phi_range=(0.4, 0.6),
        a_ii_range=(0.4, 0.6),
        steps=20
    )
    
    # =============================
    validate_exact_mode(model, params, n_sims=30)

    diagnose_all_estimators(model, params, n_runs=30)

    results = run_evaluation(model, params, n_sims=30)
    print_summary_table(results)
    print_parameter_importance(results)
    
    # Generate LaTeX tables
    generate_missing_params_table(results)
    generate_parameter_importance_table(results)
    
    # Plots
    plot_pi_curves(results)
    plot_parameter_importance(results)
    
    print(f"\nAll outputs saved to: {results_dir}")
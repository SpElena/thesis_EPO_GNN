#### Chapter 1: Binary EPO model implementation
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


### Containers
@dataclass
class EPO_Params:
    """Here we store the parameters for simulation"""
    n: int = 30 #number of nodes
    p: float = 0.5 #probability to form edges in the network
    steps: int = 20 #max number of simulation steps
    lambda_range: Tuple[float,float] = (0,1) # susceptibility
    phi_range: Tuple[float,float] = (0,1) #resilience
    a_ii_range: Tuple[float,float] = (0,1) #self-confidence
    init_op_ratio: float = 0.5 #proportion of +1 opinions
    weight_range: Tuple[float,float] = (0,1) #random edge weights, we'll stick with undirected for simplicity
    threshold_x: float = 0 #for private
    threshold_y: float = 0 #for expressed
    seed: Optional[int] = None


@dataclass
class Sim_results:
    """Storage for our simulation results"""
    graph: nx.Graph #our final graph
    history: List[dict] #states at each step
    convergence_step: int
    private_final: List[int]
    expressed_final: List[int]


### Functions for running simulations

def create_initial_graph(params: EPO_Params):
    """Create a random Erdos-Renyi graph, assign weights and parameters"""
    if params.seed is not None:
        np.random.seed(params.seed) #seed for all np.random functions

    #Create graph
    G = nx.erdos_renyi_graph(params.n, params.p, seed=params.seed)

    #Assign weights to edges
    for u,v in G.edges():
        G.edges[u,v]['weight'] = np.random.uniform(*params.weight_range)
    
    #Create and assign opinions
    n_pos = int(params.n*params.init_op_ratio)
    opinions = [1]*n_pos + [-1]*(params.n - n_pos)
    np.random.shuffle(opinions)

    for node, op in zip(G.nodes(), opinions):
        G.nodes[node].update({
            'initial_op': op,
            'private_op': op,
            'expressed_op': op,
            'lambda': np.random.uniform(*params.lambda_range),
            'phi': np.random.uniform(*params.phi_range),
            'a_ii': np.random.uniform(*params.a_ii_range)
        })

    return G


def perceived_norm(G:nx.Graph, node: int):
    """Calculate perceived social norm for given node, i.e. m_N(i)
    Defined as a weighted majority of neighbours' expressed opinions (see thesis)"""
    pos_weight = 0
    neg_weight = 0

    for neigh in G.neighbors(node):
        w = G.edges[node, neigh]['weight']
        if G.nodes[neigh]['expressed_op'] == 1:
            pos_weight += w
        else:
            neg_weight += w
    if pos_weight > neg_weight:
        return 1
    elif pos_weight < neg_weight:
        return -1
    else:
        return G.nodes[node]['expressed_op']
    

def private_upd(G: nx.Graph, node: int, params: EPO_Params):
    """Function to update private opinion of a given node
    Formula: x_i(t+1) = λ_i [a_ii * x_i(t) + (1 - a_ii) * m_N(i)] + (1 - λ_i) * x_i(0)
    See thesis for details. 
    Returns the new private opinion"""

    #Get current values for given node to match the formula
    xi = G.nodes[node]['private_op']
    x0 = G.nodes[node]['initial_op']
    lam = G.nodes[node]['lambda']
    aii = G.nodes[node]['a_ii']
    mN = perceived_norm(G,node)

    #Continous update as per EPO formulation
    x_tilde = lam*(aii*xi + (1-aii)*mN) + (1-lam)*x0

    #Compare to threshold for binary outcomes
    if x_tilde > params.threshold_x:
        return 1
    elif x_tilde < -params.threshold_x:
        return -1
    else: 
        return xi
    

def expressed_upd(G: nx.Graph, node: int, new_private: int, params: EPO_Params):
    """Function to update expressed opinion of a given node
    Formula: y_i(t+1) = φ_i * x_i(t+1) + (1 - φ_i) * m_N(i)
    See thesis for details.
    Returns the new expressed opinion """
    #Get current values for given node to match the formula
    phi = G.nodes[node]['phi']
    mN = perceived_norm(G,node)

    #Continous update as per EPO formulation
    y_tilde = phi*new_private + (1-phi)*mN

    #Compare to threshold for binary outcomes
    if y_tilde > params.threshold_y:
        return 1
    elif y_tilde < -params.threshold_y:
        return -1
    else:
        return G.nodes[node]['expressed_op']
    
def true_norm(G: nx.Graph, node: int):
    """Unweighted majority of private opinions of neighbours"""
    pos = 0
    neg = 0

    for neigh in G.neighbors(node):
        if G.nodes[neigh]['private_op'] == 1:
            pos += 1
        else:
            neg += 1
    
    if pos > neg:
        return 1
    elif pos < neg:
        return -1
    else:
        return G.nodes[node]['private_op']

def pi_node(G: nx.Graph, node: int):
    """Check if the given node experiences PI"""
    xi = G.nodes[node]['private_op']
    perceived = perceived_norm(G, node)
    true = true_norm(G, node)

    return (xi != perceived) and (perceived != true)

def net_pi(G: nx.Graph):
    """Compute the proportion of nodes experiencing PI in the network"""
    pi_count = 0
    for node in G.nodes():
        if pi_node(G, node):
            pi_count += 1

    return pi_count/len(G.nodes())


### Simulation function itself

def  run_sim(params: EPO_Params):
    """Run one full EPO simulation
    Ensure to run the necessary functions defined above prior to this one"""

    # Initialization
    G = create_initial_graph(params)

    # Run dynamics
    history = [] #list storage for state at each time step, element = dictionary with step, private, expressed

    for step in range(params.steps):
        nodes = list(G.nodes())

        changed = False #for convergence tracking
        updates = {} #dictionary to store new opinions
        #note - if we don't store before applying, we'll get other nodes seeing the current step update 

        # Compute all updates
        for node in nodes:
            new_private = private_upd(G, node, params)
            new_expressed = expressed_upd(G, node, new_private, params)

            if new_private != G.nodes[node]['private_op'] or new_expressed != G.nodes[node]['expressed_op']:
                changed = True

            updates[node] = (new_private, new_expressed)

        # Apply updates
        for node, (new_private,new_expressed) in updates.items():
            G.nodes[node]['private_op'] = new_private
            G.nodes[node]['expressed_op'] = new_expressed

        # Record current state
        private_state = [G.nodes[n]['private_op'] for n in sorted(G.nodes())]
        expressed_state = [G.nodes[n]['expressed_op'] for n in sorted(G.nodes())]

        pi = net_pi(G) #record current network pi proportion

        history.append({
            'step': step,
            'private': private_state,
            'expressed': expressed_state,
            'pi': pi
        })

        # Check if converged and stop if did
        if not changed:
            break

    # Final states
    private_final = [G.nodes[n]['private_op'] for n in sorted(G.nodes())]
    expressed_final = [G.nodes[n]['expressed_op'] for n in sorted(G.nodes())]

    return Sim_results(
        graph=G,
        history=history,
        convergence_step=len(history)-1,
        private_final=private_final,
        expressed_final=expressed_final
    )


### Simulations (generate data)

def gen_sim(params: EPO_Params, n_simulations: int = 100):
    """Run several simulations and collect their results"""
    results =[]
    for i in range(n_simulations):
        params.seed = i #different seed for each simulation
        result = run_sim(params)
        results.append(result)

        #track all is working every 10 simulations
        #if (i+1)%10 == 0:
        #   print(f"Completed {i+1}/{n_simulations} simulations")

    return results




### Running simulation

if __name__ == "__main__":
    # Single simulation test
    params = EPO_Params(seed=42)
    result = run_sim(params)
    
    print("=" * 50)
    print("EPO SIMULATION RESULTS")
    print("=" * 50)
    print(f"Converged in: {result.convergence_step} steps")
    print(f"Private: +1: {result.private_final.count(1)}, -1: {result.private_final.count(-1)}")
    print(f"Expressed: +1: {result.expressed_final.count(1)}, -1: {result.expressed_final.count(-1)}")
    mismatch = sum(1 for p, e in zip(result.private_final, result.expressed_final) if p != e)
    print(f"Nodes with private ≠ expressed: {mismatch}")
    
    # Generate training data (MAIN SIMULATIONS DATA)
    print("\n" + "=" * 50)
    print("GENERATING DATA")
    print("=" * 50)
    params = EPO_Params(seed=0)  # Start seed at 0
    all_results = gen_sim(params, n_simulations=50)


    # Check first simulation's convergence
    print(f"First simulation converged in: {all_results[0].convergence_step} steps")



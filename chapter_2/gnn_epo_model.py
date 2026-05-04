"""This is the code for GNN-EPO - the main contribution of the thesis.

The model is a surrogate of original epo model from Chapter 1
With additional ability to learn missing parameters if such need arises.
I.e. the main purpose is to evaluate the ability of GNN to perfrom according to EPO model exactly (given the same parameters)
And to evaluate the ability of GNN with epo structure to produce results in cases of missing parameters
(with futhre possibility to predict those parameters from the data)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data #PyG data object with graph structure
from typing import Optional, Dict
from torch_scatter import scatter_sum
from torch_geometric.utils import to_undirected #needed as Nx transformed to PyG gets only one direction

### First we define the class to help us within the model when parameter(s) missing
class ParameterEstimator(nn.Module):
    """Learnable function to estimate one missing parameter from node's given features and network
    (note: we'll call it node_context from here on)
    To be applied to missing lambda, phi, aii"""
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), #first layer
            nn.ReLU(), #add non-linearity
            nn.Linear(hidden_dim, 1), #output layer
            nn.Sigmoid() #output is a number in the range [0,1]
        )

    def forward(self, node_context):
        return self.net(node_context).squeeze(-1) #we squeeze for convenicence to make our tensor 1D (vector)


### Now we define the MAIN MODEL

class EPO_GNN(nn.Module):
    """
    This model is a true EPO surrogate to match our original EPO from Chapter 1
    It has two conceptual modes - Exact (all parameters provided) and Learnable (one or more missing)
    Provided the same EPO_Params as original EPO, it will create similar simulated data
    Doesn't have any learnable functions outside the potentional parameter learning for missing lambda, phi, aii
    """
    # ==================================
    ### The following section is for LEARNABLE MODE only
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder for node state (to be used in helper f'n for compute_context)
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), #takes initial_op, private_op, expressed_op of a node
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) #output is 32 hidden representations
        )

        # gnn layer for parameter estimation context (used in compute_context as well)
        self.conv = GCNConv(hidden_dim, hidden_dim) #get info of only immediate neighbours, output is [num_nodes, 32]

        #estimators for each missing parameter
        self.lambda_estimator = ParameterEstimator(hidden_dim)
        self.phi_estimator = ParameterEstimator(hidden_dim)
        self.aii_estimator = ParameterEstimator(hidden_dim)

    # ==================================
    ### FORWARD (for training) ###

    def forward(self, data: Data, mask: Optional[Dict[str, bool]] = None):
        """Function for a single synchronous step. 
        Returns continuous values for priv and expr ops (x_tilde, y_tilde) during training for better training
        We'll turn it to binary during the evaluation stage"""
        x_curr = data.x[:, 0] #vector of priv ops
        y_curr = data.x[:, 1] #of expr ops
        x0 = data.x[:, 2] #of initial ops

        #get true or estimated params values
        if mask is None:
            lam, phi, aii = data.x[:, 3], data.x[:, 4], data.x[:, 5] #get all values true
        else:
            context = self._compute_context(data.x, data)
            lam = data.x[:, 3] if mask.get('lambda', True) else self.lambda_estimator(context)
            phi = data.x[:, 4] if mask.get('phi', True) else self.phi_estimator(context)
            aii = data.x[:, 5] if mask.get('aii', True) else self.aii_estimator(context)
        
        m_N = self._compute_m_N(data, y_curr) # vector of perceived norms
        
        #calculate next step opinions (vectors)
        x_tilde = lam * (aii * x_curr + (1 - aii) * m_N) + (1 - lam) * x0
        y_tilde = phi * x_tilde + (1 - phi) * m_N
        
        #if self.training: #check if our model is in .train() mode
        return x_tilde, y_tilde #in training we return continuous ops
        #else:
        #    x_next = torch.where(x_tilde > 0, 1,torch.where(x_tilde < 0, -1, x_curr))
        #    y_next = torch.where(y_tilde > 0, 1,torch.where(y_tilde < 0, -1, y_curr))
            #return x_next, y_next #return binary opinions

    # ==================================
    ### MAIN SIMULATION ###

    def simulate(self, data: Data, steps: int, 
                 mask: Optional[Dict[str, bool]] = None):       
        """
        This piece runs full synchronous simulation matching original EPO from Ch.1
        
        Args:
            data: PyG Data is our given initial graph 
            (= to result of create_initial_graph from Ch.1 converted with graph_to_data)
            steps: max number of steps in simulation
            mask: option to mask some parameters. i.e.
                True = use provided parameter
                False = parameter is missing
                None = our exact mode (no masking)
                #e.g. {'lambda': True, 'phi': False, 'aii': True}. 
        
        Return:
            final_x: is a 2D matrix of final state, i.e. tensor with shape (num_nodes, 6) 
            history: list of state tensors at each step (for PI curves)
        """
        x = data.x.clone() #create a copy to keep original untouched
        history = []
        
        # Main simulation loop
        for step in range(steps):

            # Here we pre-compute context for parameter estimation if any masked
            #note here that we can get the estimated parameter potentially changed at each step
            #but if estimation is correct it should be the same
            if mask is not None:
                context = self._compute_context(x, data) #helper function for cleanliness
                lam_all = self.lambda_estimator(context) #get estimation vectors for a parameter
                phi_all = self.phi_estimator(context)
                aii_all = self.aii_estimator(context)

            y_curr = x[:, 1] # extract current expressed ops
            m_N = self._compute_m_N(data, y_curr) #compute current perceived soc norm per node
            
            #we'll do synchronous update (same as original epo) so store all same type updates in one list
            updates_private = []
            updates_expressed = []
            changed = False #convergence detection
            
            for node in range(len(x)):
                xi = x[node, 0].item() #current priv op for given node
                yi = x[node, 1].item() #current expr op
                x0 = x[node, 2].item() # initial op
                #note: .item() is to convert 0D tensor to regular python .float() for computations during epo eqs
                
                # Get psych parameters (true or estimated) for a given node
                if mask is None:
                    lam = x[node, 3].item()
                    phi = x[node, 4].item()
                    aii = x[node, 5].item()
                else:
                    lam = x[node, 3].item() if mask.get('lambda', True) else lam_all[node].item()
                    phi = x[node, 4].item() if mask.get('phi', True) else phi_all[node].item()
                    aii = x[node, 5].item() if mask.get('aii', True) else aii_all[node].item()
                
                m_N_val = m_N[node].item() #get the perceived norm for this given node
                
                # Private op update for a given node
                x_tilde = lam * (aii * xi + (1 - aii) * m_N_val) + (1 - lam) * x0
                x_new = 1 if x_tilde > 0 else (-1 if x_tilde < 0 else xi)
                
                # Expressed op update for a given node
                y_tilde = phi * x_new + (1 - phi) * m_N_val
                y_new = 1 if y_tilde > 0 else (-1 if y_tilde < 0 else yi)
                
                if x_new != xi or y_new != yi: #check for convergence
                    changed = True
                
                updates_private.append(x_new)
                updates_expressed.append(y_new)
            
            # Apply all updates for all nodes simultaneously
            for node in range(len(x)):
                x[node, 0] = updates_private[node]
                x[node, 1] = updates_expressed[node]
            
            history.append(x.clone())
            
            if not changed:
                break
        
        return x, history



    # ==================================
    ### HELPER FUNCTIONS ###

    def _compute_context(self, x, data):
        """Compute context (node's features & neighbourhood) for parameter's estimation (when its missing)
        x is the matrix of all nodes' private, expressed, initial opinions and 3 psych parameters (so Nx6)
        data is for graph info (edges and weights)
        Note: for simplicity we'll ignore if other psych params are present, but in the future it can be used
        to increase the model's performance"""
        state = x[:, :3] #putting all 3 ops types into a 2D tensor (matrix of Nx3)
        h = self.encoder(state) #do nn as per above self.enconder def'n (so becomes Nx32)
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=len(h))
        h_neigh = torch.relu(self.conv(h, edge_index, edge_weight)) #gnn layer to aggregate neigbhours' as defined by self.conv above
        return h + h_neigh #combine personal & neighb info
    
    def _compute_m_N(self, data: Data, y_curr: torch.Tensor):
        """Compute the same perceived norm as Chapter 1"""
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        num_nodes = len(y_curr)
        
        # make graph undirected (so each edge appears both ways), 
        # since pyg (torch.tensor) tranformation from nx made it one-directional
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=num_nodes)
        row, col = edge_index #separate source node from its neighbours
        
        # count weighted pos and neg ops of neighbours
        pos_sum = scatter_sum(edge_weight * (y_curr[col] == 1).float(), row, dim_size=num_nodes) #sum values by source node
        neg_sum = scatter_sum(edge_weight * (y_curr[col] == -1).float(), row, dim_size=num_nodes)
        
        # get the perceived norm for each node at once (results in vector)
        m_N = y_curr.clone() #assume your own opinion is majority for all nodes
        m_N = torch.where(pos_sum > neg_sum, 1, m_N) #norm for a node becomes 1 when majority is positive
        m_N = torch.where(neg_sum > pos_sum, -1, m_N) #when majority is negative
        
        return m_N

"""
This code is for traininng the EPO-GNN (on data simulated by EPO model from Chapter 1)
Required only when parameter estimators are missing (i.e. the training allows the model to learn functions
to approximate them). If all parameters are given, the training plays no role.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_1.epo_model import EPO_Params, run_sim
from gnn_epo_model import EPO_GNN

def generate_dataset(n_graphs=500):
    """Generate training data from Chapter 1 EPO model"""
    all_data = [] #storage for our training data

    #in this loop we we create a random graph with different parameters, run epo simulation on it & convert results to PyG
    for seed in range(n_graphs):
        params = EPO_Params(
            n=np.random.randint(20, 50),
            p=np.random.uniform(0.2, 0.8),
            lambda_range=(np.random.uniform(0, 0.5), np.random.uniform(0.5, 1.0)),
            phi_range=(np.random.uniform(0, 0.5), np.random.uniform(0.5, 1.0)),
            a_ii_range=(np.random.uniform(0, 0.5), np.random.uniform(0.5, 1.0)),
            steps=20,
            seed=seed
        )
        sim = run_sim(params) #run a simulation
        G = sim.graph #get the final nx graph
        history = sim.history #storing states of each steps (i.e. private op, expressed, PI)
        
        #get structure to convert to PyG from nx (required for gnn)
        nodes = sorted(G.nodes())
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous() #get matrix of [2,num of edges]
        edge_weight = torch.tensor([G.edges[u, v]['weight'] for u, v in G.edges()], dtype=torch.float) #get vector [num of edges]
        
        #now we loop through all time-steps for a given simulation & convert data to PyG
        for t in range(len(history) - 1):
            st_cur, st_next = history[t], history[t+1]
            x_t = torch.zeros(len(nodes), 6) #features (i.e private, expressed, initial, lambda, phi, aii)
            y_priv = torch.zeros(len(nodes))
            y_expr = torch.zeros(len(nodes))
            
            for i, n in enumerate(nodes):
                x_t[i, 0] = st_cur['private'][i]
                x_t[i, 1] = st_cur['expressed'][i]
                x_t[i, 2] = G.nodes[n]['initial_op']
                x_t[i, 3] = G.nodes[n]['lambda'] #true
                x_t[i, 4] = G.nodes[n]['phi'] #true
                x_t[i, 5] = G.nodes[n]['a_ii'] #true
                y_priv[i] = st_next['private'][i]
                y_expr[i] = st_next['expressed'][i]
            
            #create a PyG data object with those features and targets
            data = Data(x=x_t, edge_index=edge_index, edge_weight=edge_weight, y_private=y_priv, y_expressed=y_expr)
            all_data.append(data) #add to training dataset
        
        if (seed + 1) % 50 == 0:
            print(f"Generated {seed + 1}/{n_graphs} graphs") #just a check for progress
    
    return all_data


def train(model, train_loader, val_loader, epochs=100):
    """Training loop for our EPO-GNN model to estimate the missing parameters
    model: our EPO_GNN,
    train_loader: training data
    val_loader: validation data
    epochs: number of epochs to run"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # updates all learnable parameters in the model
    criterion = nn.MSELoss()  # since we have binary, this will fit
    best_loss = float('inf')  # initialize best validations as infinity
    
    # Validation scenarios - we test all combinations of missing parameters
    val_scenarios = [
        {'lambda': False, 'phi': True, 'aii': True},   # missing λ only
        {'lambda': True, 'phi': False, 'aii': True},   # missing φ only
        {'lambda': True, 'phi': True, 'aii': False},   # missing a_ii only
        {'lambda': False, 'phi': False, 'aii': True},  # missing λ, φ
        {'lambda': False, 'phi': True, 'aii': False},  # missing λ, a_ii
        {'lambda': True, 'phi': False, 'aii': False},  # missing φ, a_ii
        {'lambda': False, 'phi': False, 'aii': False}, # missing all
    ]
    
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0
        train_acc = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Randomly mask parameters (curriculum learning)
            mask_prob = min(0.5, epoch / epochs)
            mask = {
                'lambda': torch.rand(1).item() > mask_prob,
                'phi': torch.rand(1).item() > mask_prob,
                'aii': torch.rand(1).item() > mask_prob
            }
            
            # Ensure at least one parameter is missing
            if all(mask.values()):
                keys = ['lambda', 'phi', 'aii']
                mask[keys[torch.randint(0, 3, (1,)).item()]] = False
            
            x_pred, y_pred = model(batch, mask=mask)
            loss = criterion(x_pred, batch.y_private) + criterion(y_pred, batch.y_expressed)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Training accuracy
            x_bin = torch.where(x_pred > 0, 1, -1)
            y_bin = torch.where(y_pred > 0, 1, -1)
            acc = ((x_bin == batch.y_private).float().mean() + (y_bin == batch.y_expressed).float().mean()) / 2
            train_acc += acc.item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # validation
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0
                batch_acc = 0
                
                for mask in val_scenarios:
                    x_pred, y_pred = model(batch, mask=mask)
                    batch_loss += (criterion(x_pred, batch.y_private) + criterion(y_pred, batch.y_expressed)).item()
                    
                    x_bin = torch.where(x_pred > 0, 1, -1)
                    y_bin = torch.where(y_pred > 0, 1, -1)
                    acc = ((x_bin == batch.y_private).float().mean() + (y_bin == batch.y_expressed).float().mean()) / 2
                    batch_acc += acc.item() 
                
                val_loss += batch_loss / len(val_scenarios)
                val_acc += batch_acc / len(val_scenarios)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        #print results & save
        print(f"Epoch {epoch:3d} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'trained_EPO_GNN.pth')
    
    print("Training complete. Model saved as trained_EPO_GNN.pth")



### Additionally, we'll create a function to convert the nx graph to pyg for using in testing epo_gnn on new graphs

def graph_to_data(G, private, expressed):
    """Convert a nx graph and current opinions to PyG Data format"""
    nodes = sorted(G.nodes())
    num_nodes = len(nodes)
    
    # Node features: [private, expressed, initial, lambda, phi, a_ii]
    x = torch.zeros(num_nodes, 6)
    for i, node in enumerate(nodes):
        x[i, 0] = private[i]
        x[i, 1] = expressed[i]
        x[i, 2] = G.nodes[node]['initial_op']
        x[i, 3] = G.nodes[node]['lambda']
        x[i, 4] = G.nodes[node]['phi']
        x[i, 5] = G.nodes[node]['a_ii']
    
    # edge info
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([G.edges[u, v]['weight'] for u, v in G.edges()], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


if __name__ == "__main__":
    print("Generating data...")
    dataset = generate_dataset(n_graphs=500)
    
    np.random.seed(42)
    np.random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:split], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[split:], batch_size=32)
    
    model = EPO_GNN(hidden_dim=32)
    
    train(model, train_loader, val_loader, epochs=100)
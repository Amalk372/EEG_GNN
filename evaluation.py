import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# Assuming you have a function to load and construct graphs
def load_features_and_construct_graphs(feature_dir, labels):
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.npy')]
    
    if len(feature_files) != len(labels):
        raise ValueError("The number of labels does not match the number of feature files.")
    
    graphs = []
    
    for i, feature_file in enumerate(feature_files):
        features = np.load(os.path.join(feature_dir, feature_file))
        node_features = torch.tensor(features, dtype=torch.float).unsqueeze(1)
        edge_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long)
        
        # Add label to each graph
        label = torch.tensor([labels[i]], dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_indices.t().contiguous(), y=label)
        graphs.append(data)
    
    return graphs

# Directory containing feature files and corresponding labels
feature_dir = r'C:\Users\amalk\Desktop\Main project\new\features'
labels = np.load(r'C:\Users\amalk\Desktop\Main project\new\labels.npy')

# Load features and construct graphs with labels
graphs = load_features_and_construct_graphs(feature_dir, labels)

# Create a DataLoader for the training set
loader = DataLoader(graphs, batch_size=32, shuffle=True)

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 2)  # Assuming binary classification

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling layer to get graph-level representations
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Initialize the model and optimizer
model = GNNModel(in_channels=1, out_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(10):  # Example number of epochs
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Model training completed!")

# Save the model
torch.save(model.state_dict(), 'gnn_model.pth')

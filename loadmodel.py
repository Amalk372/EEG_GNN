import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool  # Import GCNConv and global_mean_pool
import torch.nn.functional as F

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 2)  # Adjust based on your requirements

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Load the trained model state
model = GNNModel(in_channels=1, out_channels=16)
model.load_state_dict(torch.load('gnn_model.pth', weights_only=True))
model.eval()

# Example test data
test_features = np.load('test2.npy')  # Example feature file for test data
test_node_features = torch.tensor(test_features, dtype=torch.float).unsqueeze(1)
test_edge_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long)  # Adjust as necessary

# Create a Data object
test_data = Data(x=test_node_features, edge_index=test_edge_indices.t().contiguous())

# Make predictions
with torch.no_grad():
    out = model(test_data)
    preds = out.argmax(dim=1)

# Interpret the prediction
prediction_label = "MDD" if preds.item() == 1 else "non-MDD"
print(f'Prediction for test2.npy: {prediction_label}')

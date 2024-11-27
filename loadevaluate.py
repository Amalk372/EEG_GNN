import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # Ensure DataLoader is imported
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Directory containing validation feature files and corresponding labels
validation_feature_dir = r'C:\Users\amalk\Desktop\Main project\new\features'
validation_labels = np.load(r'C:\Users\amalk\Desktop\Main project\new\labels.npy')  # Correct path for your labels

# Load validation features and construct graphs with labels
validation_graphs = load_features_and_construct_graphs(validation_feature_dir, validation_labels)

# Create a DataLoader for the validation set
validation_loader = DataLoader(validation_graphs, batch_size=32, shuffle=False)

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

# Initialize the model and load the trained state
model = GNNModel(in_channels=1, out_channels=16)  # Adjust based on your requirements
model.load_state_dict(torch.load('gnn_model.pth', weights_only=True))  # Load the trained model
model.eval()

# Evaluate the model
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in validation_loader:
        out = model(batch)
        preds = out.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

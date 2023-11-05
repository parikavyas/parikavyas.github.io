---
layout: post
title: Movie Recommendation System using Graph Representation Learning
subtitle: Building state of art Recommendation System using Graph Neural Networks
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/gnn.png
share-img: /assets/img/gnn.jpg
tags: [recommendation_systems, graph_models, graph_neural_networks, data_science, machine_learning]
author: Parika M B Vyas
---

In the digital age, the sheer volume of information and choices at our fingertips can be overwhelming. From online shopping to content streaming, the need for personalized recommendations has never been more critical. Recommendation systems are the guiding hands that help users discover products, services, or content that align with their preferences, thereby enhancing their user experience.

While recommendation systems come in various flavors, one approach that has gained significant attention and demonstrated remarkable capabilities is Graph Neural Networks (GNNs). GNNs, initially designed for graph-based data, have found a natural home in recommendation systems due to their ability to model complex relationships between users, items, and auxiliary data. In this article, we embark on a journey to explore the power of GNNs in the context of movie recommendations.

### The Power of GNNs in Recommendation Systems

Traditional collaborative filtering and matrix factorization-based recommendation systems have been the workhorses of the industry for years. However, they often face challenges when handling sparse data and capturing intricate user-item interactions. This is where GNNs step in, offering a more expressive and flexible approach to modeling recommendation scenarios.

GNNs excel at transforming recommendation problems into graph-based structures, where users and items become nodes, and interactions between them form edges. This graph-centric paradigm empowers GNNs to harness the rich latent information embedded within the connections. By propagating information across the graph, GNNs have the potential to uncover hidden patterns and relationships, ultimately delivering more accurate and 

### Implementation 

We would be using the movielens 100K dataset to build the movie recommendation system. 

Code to load and preproccess the movielens datset. Building a graph using the user-item data.

```javascript

import pandas as pd
import dgl
import torch

# Load the MovieLens 100K dataset
data_path = 'path_to_your_movielens_dataset/ml-100k/u.data'
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(data_path, sep='\t', names=column_names)

# Display the first few rows of the dataset
print("Sample data:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Data Cleaning and Conversion
# In the MovieLens dataset, the 'item_id' represents movie IDs. We can consider it as nodes for our graph.
# We'll also create a 'rating' attribute to be used as edge weights in the graph.

# Drop the 'timestamp' column, which is not needed for this analysis
df = df.drop(columns=['timestamp'])

# Ensure data types are appropriate
df['user_id'] = df['user_id'].astype(str)
df['item_id'] = df['item_id'].astype(str)

# Create a graph using deepgraphlibrary
# Each row represents a user-item interaction
# Create a 'user' node type and an 'item' node type
# Create edges connecting users to items, and use 'rating' as edge weights

# Create an empty graph
G = dgl.DGLGraph()

# Add user nodes
user_nodes = df['user_id'].unique()
G.add_nodes(len(user_nodes), {'user': torch.tensor(user_nodes)})

# Add item nodes
item_nodes = df['item_id'].unique()
G.add_nodes(len(item_nodes), {'item': torch.tensor(item_nodes)})

# Create a list of source and destination nodes for edges
src_nodes = df['user_id'].values
dst_nodes = df['item_id'].values

# Add edges (user-item interactions)
G.add_edges(src_nodes, dst_nodes, {'rating': torch.tensor(df['rating'].values)})

# Check if the graph is created successfully
print(f"\nGraph nodes: {G.number_of_nodes()}, Graph edges: {G.number_of_edges()}")

# Now, you have converted the MovieLens 100K dataset into a graph format suitable for GNNs using deepgraphlibrary.

# Additional EDA and analysis can be performed to gain further insights and to build the GNN model.

# Example:
# Get user-specific data
user_data = df.groupby('user_id').agg({'rating': ['count', 'mean']})
user_data.columns = ['rating_count', 'mean_rating']
print("\nUser-Specific Data:")
print(user_data.head())

# Get item-specific data
item_data = df.groupby('item_id').agg({'rating': ['count', 'mean']})
item_data.columns = ['rating_count', 'mean_rating']
print("\nItem-Specific Data:")
print(item_data.head())


```

Code to build a graph convolutional network on the movielens dataset using the user-movie graph.

```javascript
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import GraphConv
from sklearn.model_selection import train_test_split

# Define a simple Graph Convolutional Network (GCN) model
class GCNRecommendation(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCNRecommendation, self).__init()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x)
        return x

# Split the dataset into train and test sets
train_frac = 0.8
train_size = int(len(df) * train_frac)

train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Create a DGL Graph for the train_data
G_train = dgl.DGLGraph()
G_train.add_nodes(len(user_nodes) + len(item_nodes))
src_nodes_train = train_data['user_id'].values
dst_nodes_train = train_data['item_id'].values
G_train.add_edges(src_nodes_train, dst_nodes_train)
G_train.ndata['user'] = torch.tensor(user_nodes + item_nodes)
G_train.ndata['item'] = torch.tensor(item_nodes + user_nodes)
G_train.edata['rating'] = torch.tensor(train_data['rating'].values)

# Initialize and train the GCN model
model = GCNRecommendation(in_feats=1, hidden_size=64, out_feats=32)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    model.train()
    logits = model(G_train, G_train.edata['rating'].view(-1, 1).float())
    loss = criterion(logits.view(-1), G_train.edata['rating'].float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Create a DGL Graph for the test_data
G_test = dgl.DGLGraph()
G_test.add_nodes(len(user_nodes) + len(item_nodes))
src_nodes_test = test_data['user_id'].values
dst_nodes_test = test_data['item_id'].values
G_test.add_edges(src_nodes_test, dst_nodes_test)
G_test.ndata['user'] = torch.tensor(user_nodes + item_nodes)
G_test.ndata['item'] = torch.tensor(item_nodes + user_nodes)

# Predict ratings for the test set
model.eval()
with torch.no_grad():
    predicted_ratings = model(G_test, G_test.edata['rating'].view(-1, 1).float())

# Evaluate the recommendation system (e.g., using RMSE or other relevant metrics)
from sklearn.metrics import mean_squared_error
import numpy as np

true_ratings = test_data['rating'].values
predicted_ratings = predicted_ratings.cpu().numpy().flatten()
rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
print(f'Root Mean Squared Error (RMSE) on the test set: {rmse}')


```














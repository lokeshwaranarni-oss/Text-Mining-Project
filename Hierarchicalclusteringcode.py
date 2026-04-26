# Hierarchical Clustering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

# Load Dataset
data = pd.read_csv(
    "Hierarchicalclustering_Food Waste Dataset from a University Canteen.csv"
)

# Use only numeric columns
data = data.select_dtypes(
    include=['int64', 'float64']
)

# Standardization
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

# Dendrogram

plt.figure(figsize=(10,6))

linked = linkage(
    scaled_data,
    method='ward'
)

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Data Points")

plt.ylabel("Distance")

plt.show()

# Model
model = AgglomerativeClustering(
    n_clusters=3
)

clusters = model.fit_predict(
    scaled_data
)

# Scatter Plot
plt.figure()

plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=clusters
)

plt.title("Cluster Visualization")

plt.xlabel(data.columns[0])

plt.ylabel(data.columns[1])

plt.show()

# Cluster Distribution

plt.figure()

sns.countplot(
    x=clusters
)

plt.title("Cluster Distribution")

plt.show()

# Pairplot

data_plot = pd.DataFrame(
    scaled_data,
    columns=data.columns
)

data_plot["Cluster"] = clusters

sns.pairplot(
    data_plot,
    hue="Cluster"
)

plt.show()
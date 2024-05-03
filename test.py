import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from numpy import unique
import matplotlib.pyplot as plt

st.title('Explore Clustering Algorithms on Different Datasets')

st.write("""
         # Explore **clustering** algorithms on different datasets
         """)

dataset_name = st.sidebar.selectbox('Choose your desired dataset please: ',
                                    ('Aggregation', 'Blobs', 'Circles', 'Moons'))
st.write(f'Selected dataset: {dataset_name}')

def load_dataset(dataset_name):
    if dataset_name == 'Aggregation':
        X, y = make_blobs(n_samples=1000, centers=4, random_state=42)
    elif dataset_name == 'Blobs':
        X, y = make_blobs(n_samples=1000, centers=5, random_state=42)
    elif dataset_name == 'Circles':
        X, y = make_circles(n_samples=1000, noise=0.05, random_state=42)
    elif dataset_name == 'Moons':
        X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
    return X, y

X, y = load_dataset(dataset_name=dataset_name)
st.write(
    f"""
    Dataset info:\n
    Shape = {X.shape}\n
    No. of clusters: {len(unique(y))}
    """
)

# Choose clustering algorithm
clustering_algorithm = st.sidebar.selectbox('Choose Clustering Algorithm: ',
                                            ('K-Means', 'Agglomerative', 'DBSCAN'))

# Adjust algorithm parameters using sliders
if clustering_algorithm == 'K-Means':
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2)
    algorithm = KMeans(n_clusters=n_clusters)
elif clustering_algorithm == 'Agglomerative':
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2)
    linkage = st.sidebar.selectbox('Linkage', ('ward', 'complete', 'average'))
    algorithm = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
elif clustering_algorithm == 'DBSCAN':
    eps = st.sidebar.slider('Epsilon', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    min_samples = st.sidebar.slider('Minimum Samples', min_value=1, max_value=10, value=5)
    algorithm = DBSCAN(eps=eps, min_samples=min_samples)

# Apply clustering algorithm
cluster_labels = algorithm.fit_predict(X)

# Plot original data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(plt)

# Plot clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title(f"Clustering Algorithm: {clustering_algorithm} - Class Assignments")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(plt)
import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_moons
from numpy import unique
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



st.title(':blue[C]:red[l]:blue[a]:red[s]:blue[s]:red[i]:blue[f]:red[i]:blue[c]:red[a]:blue[t]:red[i]:blue[o]:red[n] :sunglasses:')

st.write("""
         # Explore **classification** algorithms on different dataset
         """)

dataset_name = st.sidebar.selectbox('Choose your desired dataset please: ',
                                    ('Aggregation', 'Blobs', 'Circles', 'Moons'))

clustering_algorithm = st.sidebar.selectbox('Choose Clustering Algorithm: ',
                                            ('K-Means', 'Agglomerative', 'DBSCAN'))
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

# Apply clustering algorithm
clusterer = KMeans(n_clusters=len(unique(y)))
cluster_labels = clusterer.fit_predict(X)

# Display clustering results
st.write("Clustering Results:")
st.write(cluster_labels)

# Plot original data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(plt)

# Plot clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='rainbow', alpha=0.7)
plt.title("Clustering Algorithm: Class Assignments")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(plt)
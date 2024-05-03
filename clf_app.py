import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_moons
from numpy import unique
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation, MeanShift
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

class StreamlitApp:
    def __init__(self) -> None:
        self.dataset_name = st.sidebar.selectbox('Choose your desired dataset please: ',
                                            ('Aggregation', 'Blobs', 'Circles', 'Moons', 'Noisy Circles', 'Synthetic Moons', 'Synthetic Circles', 'Synthetic Blobs'))
        self.clustering_algorithm = st.sidebar.selectbox('Choose Clustering Algorithm: ',
                                                    ('K-Means', 'Agglomerative', 'DBSCAN', 'Spectral Clustering', 'Affinity Propagation', 'Mean Shift', 'Gaussian Mixture Models'))
        if self.clustering_algorithm == 'K-Means':
            self.n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2, key='kmeans')
        elif self.clustering_algorithm == 'Agglomerative':
            self.n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2, key='agg')
            self.linkage = st.sidebar.selectbox('Linkage', ('ward', 'complete', 'average'))
        elif self.clustering_algorithm == 'DBSCAN':
            self.eps = st.sidebar.slider('Epsilon', min_value=0.1, max_value=1.0, value=0.5, step=0.05, key='dbscan')
            self.min_samples = st.sidebar.slider('Minimum Samples', min_value=1, max_value=10, value=5, key='dbscan2')
        elif self.clustering_algorithm == 'Spectral Clustering':
            self.n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2, key='spectral')
        elif self.clustering_algorithm == 'Affinity Propagation':
            self.damping = st.sidebar.slider('Damping', min_value=0.5, max_value=1.0, value=0.5, step=0.1, key='affinity')
        elif self.clustering_algorithm == 'Mean Shift':
            self.bandwidth = st.sidebar.slider('Bandwidth', min_value=0.1, max_value=1.0, value=0.5, step=0.1, key='meanshift')
        elif self.clustering_algorithm == 'Gaussian Mixture Models':
            self.n_components = st.sidebar.slider('Number of Components', min_value=2, max_value=10, value=2, key='gmm')
    
    def run(self):
        self.__add_title()
        self.__load_dataset(self.dataset_name)
        self.__load_algorithm(self.clustering_algorithm)
        self.__apply_clustering()
        self.__dispaly()
    
    def __add_title(self):
        st.title(':blue[C]:red[L]:blue[U]:red[S]:blue[T]:red[E]:blue[R]:red[I]:blue[N]:red[G] :sunglasses:')
        st.write("""
                # Explore **CLUSTERING** algorithms on different dataset
                """)
        st.write(f'Selected dataset: {self.dataset_name}')
        
    def __load_dataset(self, dataset_name):
        if dataset_name == 'Aggregation':
            self.X, self.y = make_blobs(n_samples=1000, centers=4, random_state=42)
        elif dataset_name == 'Blobs':
            self.X, self.y = make_blobs(n_samples=1000, centers=5, random_state=42)
        elif dataset_name == 'Circles':
            self.X, self.y = make_circles(n_samples=1000, noise=0.05, random_state=42)
        elif dataset_name == 'Moons':
            self.X, self.y = make_moons(n_samples=1000, noise=0.05, random_state=42)
        elif dataset_name == 'Noisy Circles':
            self.X, self.y = make_circles(n_samples=1000, noise=0.03, factor=0.5, random_state=42)
        elif dataset_name == 'Synthetic Moons':
            self.X, self.y = make_moons(n_samples=1000, noise=0.03, random_state=42)
        elif dataset_name == 'Synthetic Circles':
            self.X, self.y = make_circles(n_samples=1000, noise=0.03, random_state=42)
        elif dataset_name == 'Synthetic Blobs':
            self.X, self.y = make_blobs(n_samples=1000, centers=5, cluster_std=[1.0, 2.5, 0.5, 3.0, 1.5], random_state=42)

    def __load_algorithm(self, clustering_algorithm):
        if clustering_algorithm == 'K-Means':
            self.algorithm = KMeans(n_clusters=self.n_clusters)
        elif clustering_algorithm == 'Agglomerative':
            self.algorithm = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        elif clustering_algorithm == 'DBSCAN':
            self.algorithm = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        elif clustering_algorithm == 'Spectral Clustering':
            self.algorithm = SpectralClustering(n_clusters=self.n_clusters)
        elif clustering_algorithm == 'Affinity Propagation':
            self.algorithm = AffinityPropagation(damping=self.damping)
        elif clustering_algorithm == 'Mean Shift':
            self.algorithm = MeanShift(bandwidth=self.bandwidth)
        elif clustering_algorithm == 'Gaussian Mixture Models':
            self.algorithm = GaussianMixture(n_components=self.n_components)

        return self.algorithm
    
    def __apply_clustering(self):
        self.__load_dataset(dataset_name=self.dataset_name)
        self.__load_algorithm(clustering_algorithm=self.clustering_algorithm)
        self.cluster_labels = self.algorithm.fit_predict(self.X)

    def __dispaly(self):
        plt.style.use('dark_background')
        # Plot original data
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c='blue', alpha=0.7)
        plt.title("Original Data")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        st.pyplot(plt)

        # Plot clustering results
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.cluster_labels, cmap='plasma', alpha=0.7)
        plt.title("Clustering Algorithm: Class Assignments")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        st.pyplot(plt)


if __name__ == '__main__':
    app = StreamlitApp()
    app.run()
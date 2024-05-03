import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class StreamlitApp:
    def __init__(self):
        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Iris', 'Breast Cancer', 'Wine')
        )
        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Random Forest')
        )
        self.params = self.add_parameter_ui()

    def run(self):
        self.display_title()
        self.load_dataset()
        self.train_classifier()
        self.evaluate_classifier()
        self.plot_dataset()
        

    def display_title(self):
        st.title('Streamlit Example')
        st.write("""
        # Explore different classifiers and datasets
        Which one is the best?
        """)

    def load_dataset(self):
        st.write(f"## {self.dataset_name} Dataset")
        data = self.get_dataset()
        self.X = data.data
        self.y = data.target
        st.write('Shape of dataset:', self.X.shape)
        st.write('Number of classes:', len(np.unique(self.y)))

    def get_dataset(self):
        if self.dataset_name == 'Iris':
            return datasets.load_iris()
        elif self.dataset_name == 'Wine':
            return datasets.load_wine()
        else:
            return datasets.load_breast_cancer()

    def add_parameter_ui(self):
        params = {}
        if self.classifier_name == 'SVM':
            params['C'] = st.sidebar.slider('C', 0.01, 10.0)
        elif self.classifier_name == 'KNN':
            params['K'] = st.sidebar.slider('K', 1, 15)
        else:
            params['max_depth'] = st.sidebar.slider('max_depth', 2, 15)
            params['n_estimators'] = st.sidebar.slider('n_estimators', 1, 100)
        return params

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            return SVC(C=self.params['C'])
        elif self.classifier_name == 'KNN':
            return KNeighborsClassifier(n_neighbors=self.params['K'])
        else:
            return RandomForestClassifier(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                random_state=1234
            )

    def train_classifier(self):
        clf = self.get_classifier()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=1234
        )
        clf.fit(self.X_train, self.y_train)
        self.y_pred = clf.predict(self.X_test)

    def evaluate_classifier(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)

    def plot_dataset(self):
        pca = PCA(2)
        X_projected = pca.fit_transform(self.X)
        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]
        fig = plt.figure()
        plt.scatter(x1, x2, c=self.y, alpha=0.8, cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar()
        st.pyplot(fig)

if __name__ == '__main__':
    app = StreamlitApp()
    app.run()
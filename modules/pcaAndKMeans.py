from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st

# ! PCA
def pcaFunction(PCAandKmeansData):
    pca = PCA(n_components=10)
    reduced_data = pca.fit_transform(PCAandKmeansData)
    return reduced_data

# ! K means Clustering

def kmeansFunction(reduced_data, PCAandKmeansData):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(reduced_data)

    # Obtain the cluster labels for each data point
    cluster_labels = kmeans.labels_

    reduced_data = pd.DataFrame(reduced_data, index=PCAandKmeansData.index[:])

    # Add the 'Cluster_Label' column to your DataFrame
    reduced_data['Cluster_Label'] = cluster_labels

    # Save the updated DataFrame to a new CSV file
    st.write(reduced_data)
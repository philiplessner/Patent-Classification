from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


def cluster(vectors, num_clusters=3):
    '''Perform clustering on vectorized features
       Args:
            vectorized: samples, features matrix
            num_clusters: number of KMeans clusters
       Returns:
            km: cluster number for each sample
    '''
    km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
    km.fit(vectors)
    labels = km.labels_
    return km, labels


def cluster_metrics(km, labels, vectors):
    sil_scores = metrics.silhouette_samples(vectors,
                                            labels, metric='euclidean')
    return sil_scores


def sil_plot(df):
    # fig, axes = plt.subplots(2, 1)
    plt.figure()
    df_p = df[df['clusters'] == 0]
    df_p['silhouettes'].plot(kind='barh')
    plt.xlim(0.0, 1.0)
    plt.show()

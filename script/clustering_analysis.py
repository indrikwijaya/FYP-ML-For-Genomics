import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GMM
from scipy.stats import pearsonr
from HungarianAlgorithm import *
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings("ignore")

time_series_df_file = sys.argv[1]
stem_df_file = sys.argv[2]
sig_profs = sys.argv[3]

sig_profs = [int(x) for x in sig_profs.split(",")]

'''Process STEM Clusters'''


def filter_significant_clusters_stem(stem_df_file, sig_profs):
    stem_df = pd.read_csv(stem_df_file, sep='\t')
    stem_df = stem_df.drop(['Selected', 'SPOT'], axis=1)
    stem_df = stem_df[stem_df['Profile'].isin(sig_profs)]
    stem_df = stem_df.set_index('AccNum')
    stem_df.index.name = None
    sig_clust = []
    for i in range(0, len(sig_profs)):
        cluster_i = stem_df[stem_df['Profile'] == sig_profs[i]]
        cluster_i = cluster_i[['t0', 't1', 't2', 't3', 't4', 't5', 't6']]
        cluster_i.T.plot(legend=False, figsize=(15, 7))
        plt.title('Profile %d with %d genes' % (i, len(cluster_i)), fontsize=20)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=35)
        plt.savefig('%s_Cluster_%d_%d' % ('STEM', i, len(cluster_i)))
        # plt.show()
        cluster_i_gene = cluster_i.axes[0].tolist()
        gene_file = open('STEM_cluster_'+str(i)+'.txt', 'w')
        gene_file.write('\n'.join(cluster_i_gene))
        gene_file.close()
        sig_clust.append(cluster_i_gene)

    true_cluster = []
    for i in range(0, len(sig_clust)):
        true_cluster.append([i, sig_clust[i]])

    datasize = sum(len(cluster) for cluster in sig_clust)
    stem_df = stem_df.drop(['Profile'], axis=1)
    return true_cluster, datasize, stem_df

stem_clust, datasize, stem_df = filter_significant_clusters_stem(stem_df_file, sig_profs)
opt_cluster = len(stem_clust)

'''Analyse different clustering algorithms'''
# time_series_df = stem_df
time_series_df = pd.read_csv(time_series_df_file, sep='\t')
# print(time_series_df.head())
# time_series_df = time_series_df[time_series_df.index.isin(stem_df)]
# print(time_series_df.head())

# return plot and optimal cluster


def generate_clusters(time_series_df, cluster_model, cluster_type):

    if cluster_type[:3] == 'HAC':
        cluster_labels = cluster_model.fit(time_series_df.values).labels_
    else:
        cluster_labels = cluster_model.fit(time_series_df.values).predict(time_series_df.values)

    s = pd.Series(cluster_labels)
    clusters = s.unique()
    clusters_id = []
    for i in range(0, len(clusters)):
        cluster_indices = s[s == clusters[i]].index
        gene_cluster = time_series_df.T.iloc[:, cluster_indices].transpose().index
        gene_file = open(cluster_type + '_cluster_' + str(i) + '_' + str(len(gene_cluster)) + '.txt', 'w')
        gene_file.write("\n".join(list(gene_cluster.values)))
        clusters_id.append([i, list(gene_cluster)])

        time_series_df.T.iloc[:, cluster_indices].plot(legend=False, figsize=(20, 10))
        plt.title('%d genes' % len(gene_cluster), fontsize=45)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)
        plt.savefig("%s_Cluster_%d_%d" % (cluster_type, clusters[i], len(gene_cluster)))

    return clusters_id


'''
K-means
'''
df_kmeans = KMeans(n_clusters=opt_cluster, random_state=1)
kmeans_cluster = generate_clusters(time_series_df, df_kmeans, 'KMeans')
'''
GMM
'''
df_gmm_full = GMM(n_components=opt_cluster, covariance_type='full')
gmm_full_cluster = generate_clusters(time_series_df, df_gmm_full, 'GMM-full')
print(gmm_full_cluster)
df_gmm_tied = GMM(n_components=opt_cluster, covariance_type='tied')
gmm_tied_cluster = generate_clusters(time_series_df, df_gmm_tied, 'GMM-tied')
print(gmm_tied_cluster)
df_gmm_diag = GMM(n_components=opt_cluster, covariance_type='diag')
gmm_diag_cluster = generate_clusters(time_series_df, df_gmm_diag, 'GMM-diag')
print(gmm_diag_cluster)
df_gmm_sph = GMM(n_components=opt_cluster, covariance_type='spherical')
gmm_sph_cluster = generate_clusters(time_series_df, df_gmm_sph, 'GMM-sph')
print(gmm_sph_cluster)

'''
Hierarchical clustering
'''


def pearson_affinity(M):
    return 1-np.array([[pearsonr(a, b)[0] for a in M] for b in M])

df_hac_ward = AgglomerativeClustering(n_clusters=opt_cluster, linkage='ward', affinity='euclidean')
hac_ward_cluster = generate_clusters(time_series_df, df_hac_ward, 'HAC-ward')
print(hac_ward_cluster)
df_hac_comp = AgglomerativeClustering(n_clusters=opt_cluster, linkage='complete', affinity=pearson_affinity)
hac_comp_cluster = generate_clusters(time_series_df, df_hac_comp, 'HAC-comp')
print(hac_comp_cluster)
df_hac_avg = AgglomerativeClustering(n_clusters=opt_cluster, linkage='average', affinity=pearson_affinity)
hac_avg_cluster = generate_clusters(time_series_df, df_hac_avg, 'HAC-avg')
print(hac_avg_cluster)

'''
Clustering evaluation 1: Clustering Accuracy
'''


def intersect(a, b):
    return list(set(a) & set(b))


def label_to_true(true_cluster, test_cluster):
    updated_cluster = []
    for i in range(0, len(test_cluster)):
        max_label = 0
        for j in range(0, len(true_cluster)):
            max_intersect = len(intersect(test_cluster[i][1], true_cluster[j][1]))
            if len(intersect(test_cluster[i][1], true_cluster[j][1])) > max_label:
                max_label = max_intersect
                updated_cluster.append([true_cluster[j][0], test_cluster[i][1]])

    return updated_cluster


def clustering_accuracy(true_cluster, test_cluster):
    count = 0
    for i in test_cluster:
        for j in true_cluster:
            if i[0] == j[0]:
                count += len(intersect(i[1], j[1]))
    return count/datasize

print('#############Clustering Evaluation 1: Clustering Accuracy#################')
print('Kmeans accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, kmeans_cluster))))
print('GMM_full accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, gmm_full_cluster))))
print('GMM_diag accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, gmm_diag_cluster))))
print('GMM_tied accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, gmm_tied_cluster))))
print('GMM_sph accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, gmm_sph_cluster))))
print('HAC_ward accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, hac_ward_cluster))))
print('HAC_comp accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, hac_comp_cluster))))
print('HAC_avg accuracy: ' + str(clustering_accuracy(stem_clust, label_to_true(stem_clust, hac_avg_cluster))))

'''
Clustering evaluation 2: ME distance
'''


def generate_graph(stem_clust, model_clust):
    weights_graph = {}
    for j in stem_clust:
        j_weight = {}
        for i in model_clust:
            j_weight[i[0]] = len(intersect(j[1], i[1]))
        weights_graph[j[0]] = j_weight

    weights_mat = []
    for i in weights_graph:
        w_i = weights_graph[i]
        weight = []
        for j in w_i:
            weight.append(w_i[j])
        weights_mat.append(weight)
    return weights_mat

me_kmeans = 1 - maxWeightMatching(generate_graph(stem_clust, kmeans_cluster))[2]/datasize

me_gmm_full = 1 - maxWeightMatching(generate_graph(stem_clust, gmm_full_cluster))[2]/datasize
me_gmm_tied = 1 - maxWeightMatching(generate_graph(stem_clust, gmm_tied_cluster))[2]/datasize
me_gmm_diag = 1 - maxWeightMatching(generate_graph(stem_clust, gmm_diag_cluster))[2]/datasize
me_gmm_sph = 1 - maxWeightMatching(generate_graph(stem_clust, gmm_sph_cluster))[2]/datasize

me_hac_ward = 1 - maxWeightMatching(generate_graph(stem_clust, hac_ward_cluster))[2]/datasize
me_hac_comp = 1 - maxWeightMatching(generate_graph(stem_clust, hac_comp_cluster))[2]/datasize
me_hac_avg = 1 - maxWeightMatching(generate_graph(stem_clust, hac_avg_cluster))[2]/datasize

print('#############Clustering Evaluation 2: ME Distance#################')
print('KMeans accuracy: ' + str(me_kmeans))
print('GMM_full accuracy: ' + str(me_gmm_full))
print('GMM_tied accuracy: ' + str(me_gmm_tied))
print('GMM_diag accuracy: ' + str(me_gmm_diag))
print('GMM_sph accuracy: ' + str(me_gmm_sph))
print('HAC_ward accuracy: ' + str(me_hac_ward))
print('HAC_comp accuracy: ' + str(me_hac_comp))
print('HAC_avg accuracy: ' + str(me_hac_avg))

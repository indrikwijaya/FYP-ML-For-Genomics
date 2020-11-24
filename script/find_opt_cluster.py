import pandas as pd
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans

time_series_df_file = sys.argv[1]
k_max = int(sys.argv[2])
gene_type = sys.argv[3]

time_series_df = pd.read_csv(time_series_df_file, sep='\t').iloc[:, 1:]

'''
GMM: BIC & AIC
'''


def plot_ic_curve(ic_array, cov_type, title, path):
    x = np.arange(0, len(ic_array), 1)
    plt.figure(figsize=(20, 10))
    plt.plot(x, ic_array, marker='o')
    # plt.ylabel(cov_type, fontsize=25)
    plt.xlabel("no components", fontsize=35)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    # plt.title(cov_type+'_'+title, fontsize=45)
    plt.savefig(str(gene_type)+'_'+path)



def plot_ll_pen_curve(ll, aic_pen, bic_pen, cov_type, path):
    x = np.arange(0, len(ll), 1)
    plt.figure(figsize=(20, 10))
    plt.plot(x, ll, marker='o')
    plt.plot(x, aic_pen, marker='+')
    plt.plot(x, bic_pen, marker='*')
    plt.ylabel(cov_type, fontsize=25)
    plt.xlabel("no components", fontsize=35)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.legend(['log-likelihood', 'AIC_penalty', 'BIC_penalty'],
    # bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    plt.legend(['Log-likelihood', 'AIC penalty', 'BIC penalty'], loc='upper left', fontsize=35)
    plt.title(cov_type, fontsize=45)
    plt.savefig(str(gene_type) + '_' + path)

    # plt.show()


def gmm(dataframe):

    # Number of samples per component
    n_samples = len(dataframe.iloc[:, 1])

    # Generate random sample, two components
    np.random.seed(0)

    X = dataframe.values

    lowest_bic = np.infty
    lowest_aic = np.infty

    aic_pen = {'spherical': [], 'tied': [], 'diag': [], 'full': []}
    bic_pen = {'spherical': [], 'tied': [], 'diag': [], 'full': []}
    bic = {'spherical': [], 'tied': [], 'diag': [], 'full': []}
    aic = {'spherical': [], 'tied': [], 'diag': [], 'full': []}
    log_likelihood = {'spherical': [], 'tied': [], 'diag': [], 'full': []}

    n_components_range = range(1, k_max+1)
    cv_types = ['spherical', 'tied', 'diag', 'full']

    for cv_type in cv_types:
        for n_components in n_components_range:
            fix_pen = n_components*7 + n_components-1
            penalty = {'spherical': n_components+fix_pen, 'tied': 28+fix_pen,
                       'diag': 7 * n_components+fix_pen, 'full': 28 * n_components+fix_pen}
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type,
                                          max_iter=1000)
            gmm.fit(X)
            aic_pen[cv_type].append(2*penalty[cv_type])
            bic_pen[cv_type].append(math.log2(len(X))*penalty[cv_type])

            log_likelihood[cv_type].append(gmm.lower_bound_*(-2))

            bic[cv_type].append(gmm.bic(X))
            aic[cv_type].append(gmm.aic(X))

            if bic[cv_type][-1] < lowest_bic:
                lowest_bic = bic[cv_type][-1]
                best_gmm_bic = gmm

            if aic[cv_type][-1] < lowest_aic:
                lowest_aic = aic[cv_type][-1]
                best_gmm_aic = gmm
    # print(len(log_likelihood))
    # print(len(aic))
    '''Log-likelihood Plots'''
    ll_sph = np.array(log_likelihood['spherical'])
    aic_sph = np.array(aic_pen['spherical'])
    bic_sph = np.array(bic_pen['spherical'])
    plot_ll_pen_curve(ll_sph, aic_sph, bic_sph, 'Spherical', '_ll_pen_spherical.png')
    plot_ic_curve(ll_sph, 'log-likelihood', 'Spherical', '_ll_spherical.png')

    ll_tied = np.array(log_likelihood['tied'])
    aic_tied = np.array(aic_pen['tied'])
    bic_tied = np.array(bic_pen['tied'])
    plot_ll_pen_curve(ll_tied, aic_tied, bic_tied, 'Tied', '_ll_pen_tied.png')
    plot_ic_curve(ll_tied, 'log-likelihood', 'Tied', '_ll_tied.png')

    ll_diag = np.array(log_likelihood['diag'])
    aic_diag = np.array(aic_pen['diag'])
    bic_diag = np.array(bic_pen['diag'])
    plot_ll_pen_curve(ll_diag, aic_diag, bic_diag, 'Diag', '_ll_pen_diag.png')
    plot_ic_curve(ll_diag, 'log-likelihood', 'Diag', '_ll_diag.png')

    ll_full = np.array(log_likelihood['full'])
    aic_full = np.array(aic_pen['full'])
    bic_full = np.array(bic_pen['full'])
    plot_ll_pen_curve(ll_full, aic_full, bic_full, 'Full', '_ll_pen_full.png')
    plot_ic_curve(ll_full, 'log-likelihood', 'Full', '_ll_full.png')

    '''BIC Plots'''
    bic_sph = np.array(bic['spherical'])
    plot_ic_curve(bic_sph, 'BIC', 'Spherical', '_bic_spherical.png')

    bic_tied = np.array(bic['tied'])
    plot_ic_curve(bic_tied, 'BIC', 'Tied', '_bic_tied.png')

    bic_diag = np.array(bic['diag'])
    plot_ic_curve(bic_diag, 'BIC', 'Diagonal', '_bic_diag.png')

    bic_full = np.array(bic['full'])
    plot_ic_curve(bic_full, 'BIC', 'Full', '_bic_full.png')

    '''AIC Plots'''
    aic_sph = np.array(aic['spherical'])
    plot_ic_curve(aic_sph, 'AIC', 'Spherical', '_aic_spherical.png')

    aic_tied = np.array(aic['tied'])
    plot_ic_curve(aic_tied, 'AIC', 'Tied', '_aic_tied.png')

    aic_diag = np.array(aic['diag'])
    plot_ic_curve(aic_diag, 'AIC', 'Diagonal', '_aic_diag.png')

    aic_full = np.array(aic['full'])
    plot_ic_curve(aic_full, 'AIC', 'Full', '_aic_full.png')

    print("For BIC, best GMM model:")
    print(best_gmm_bic)
    print("For AIC, best GMM model:")
    print(best_gmm_aic)

    return bic, aic


gmm_model = gmm(time_series_df)

'''
Elbow Curve
'''


def plot_elbow(gene_df, k_max, gene_type):
    distortions = []
    k_range = range(2, k_max)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(gene_df.iloc[:, 1:].values)
        distortions.append(kmeans.inertia_)
    stirling_kmeans_fig = plt.figure(figsize=(20, 10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(k_range, distortions, marker='o')

    # plt.xlabel(r'\textbf{Number of clusters} $K$', fontsize = 25)
    plt.xticks(fontsize=40)
    plt.ylabel(r'$W_k$', fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid(True)
    # plt.title('Elbow curve'+' for '+gene_type, fontsize = 35)
    plt.savefig(str(gene_type) + '_elbow.png')
    # plt.show()

plot_elbow(time_series_df, k_max, gene_type)

'''
Gap stats
'''


def optimalK(data, nrefs, maxClusters):
    """
    Calculate KMeans optimal k using Gap statistic from Tibshiarni, Walther, Hastie Params:
        data:ndarry of shape (n_samples, n_features)
        nrefs: no of sample reference datasets to create
        maxClusters: maximum no of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference for dispersion results
        refDisps = np.zeros(nrefs)

        # for n references, generate random sample and perfrm kmeans getting resulting dispersion for each loop
        for i in range(nrefs):
            # create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

k, gapdf = optimalK(time_series_df.values, nrefs=5, maxClusters=k_max)


def plot_gap(gapdf, k, gene_type):

    plt.figure(figsize=(25, 15))

    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')

    # plt.xlabel(r'\textbf{Number of clusters} $K$', fontsize = 25)
    plt.xticks(fontsize=40)
    # plt.ylabel(r'Gap', fontsize = 40)
    plt.yticks(fontsize=40)
    plt.grid(True)
    # plt.title('Gap values by cluster count', fontsize = 35)
    plt.savefig(str(gene_type)+'_gap.png')
    # plt.show()

plot_gap(gapdf, k, gene_type)

from numba import jit, prange
import pandas as pd
import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
import matplotlib as mpl
from cdlib import algorithms
from cdlib import datasets
import time
import sys

@jit(nopython=True)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, alpha, C):
    """
        A function to compute the Balanced Forman curvature (BFC) of a given NetworkX graph.
        Current version can only compute BFC for unweighted graph.

        Parameters
        ----------
        A: Adjacency matrix of a graph.
        A2: A squared.
        d_in: Row sum of A
        d_out: Column sum of A
        N: The total number of nodes in a graph.
        C: Curvature matrix that saves curvature of edge (ij) in C_ij
        """
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0: #if no edges, then bfc is zero
                C[i, j] = 0
                continue

            if d_in[i] > d_out[j]: #only for directed case
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                continue

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += alpha * (sharp_ij / (d_max * lambda_ij))


def balanced_forman_curvature(A, alpha, C=None):
    """
        A final function to compute the Balanced Forman curvature (BFC) of a given NetworkX graph.

        Parameters
        ----------
        A: Adjacency matrix of a graph.
        C: Curvature matrix that saves curvature of edge (ij) in C_ij. Default is n by n zero matrix.

        Returns
        -------
        C: Curvature matrix that saves curvature of edge (ij) in C_ij
        """
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))

    _balanced_forman_curvature(A, A2, d_in, d_out, N, alpha, C)
    return C

# %%
class LowerORicci:
    def __init__(self, G: nx.Graph):
        """
        A class to compute the Lower-Ricci curvature (LRC) of a given NetworkX graph.
        Current version can only compute LRC for unweighted and undirected graph.

        Parameters
        ----------
        G: NetworkX graph.
        """
        self.G = G.copy()

    def lower_curvature(self):
        """Compute LRC for all edges in G.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "lrc" on edges.
        """
        C = {}
        for i, j in self.G.edges:
            n_ij = len(sorted(nx.common_neighbors(self.G, i, j)))
            n_i = len(sorted(nx.all_neighbors(self.G, i)))
            n_j = len(sorted(nx.all_neighbors(self.G, j)))
            n_max = max(n_i, n_j)
            n_min = min(n_i, n_j)
            C[(i, j)] = 2/n_i + 2/n_j - 2 + 2*n_ij/n_max + n_ij/n_min
        nx.set_edge_attributes(self.G, C, "lrc")
        return self.G

    #saving the result in a dictionary
    def set_balancedforman_edge(self,C):
        """Make a dictionary called attri that saves the value of BFC for each edge
        Parameters
        ----------
        C: The curvature matrix.

        Returns
        -------
        attri: A dictionary that saves that value of BFC as value and edge as keys.
        """
        attri = {}
        nodelist = list(self.G)
        for i, j in nx.edges(self.G):
            x = nodelist.index(i)
            y = nodelist.index(j)
            attri[(i,j)] = C[x, y]
            attri[(j,i)] = C[y, x]
        return attri

    def compute_balancedformancurv(self, alpha):
        """Compute Balanced Forman Curvature of edges.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "nfc" on edges.
        """
        A = nx.to_numpy_array(self.G)
        curv = balanced_forman_curvature(A, alpha, C = None)
        a = self.set_balancedforman_edge(curv)
        nx.set_edge_attributes(self.G, a, "bfc")
        return self.G


# %%
def lowercurv(G: nx.Graph):
    lrc = LowerORicci(G)
    G = lrc.lower_curvature()
    return G

def balancedcurv(G: nx.Graph, alpha=1):
    bfc = LowerORicci(G)
    G = bfc.compute_balancedformancurv(alpha)
    return G

def remove_curv(k, G, name):
    ebunch = []
    for e in G.edges(data=True):
        if e[2][name] < k:
            ebunch.append((e[0], e[1]))
    G.remove_edges_from(ebunch)

def group_nodes_by_feature(G, feature):
    groups = {}
    for node, data in G.nodes(data=True):
        group = data.get(feature, None)
        if group is not None:
            if group not in groups:
                groups[group] = []
            groups[group].append(node)
    return groups

# %%
def estimate_thershold(lowdf, max_iter=1000, pas=0.01, lownorm='lrc', verbose=1):
  x = np.ravel(lowdf).astype(float)
  x = x.reshape(-1, 1)
  gmm2 = GMM(n_components = 2, max_iter=max_iter, random_state=10, covariance_type = 'full')

  # find useful parameters
  mean2 = gmm2.fit(x).means_
  covs2  = gmm2.fit(x).covariances_
  weights2 = gmm2.fit(x).weights_

  # find the middle point
  p = 100 #the probability of x
  x_val = -3

  sgmm = min(mean2[0][0], mean2[1][0])
  bgmm = max(mean2[0][0], mean2[1][0])

  for i in np.arange(round(float(sgmm), 2), round(float(bgmm),2)+pas, pas):
      y_0 = norm.pdf(i, float(mean2[0][0]), np.sqrt(float(covs2[0][0][0])))*weights2[0] # 1st gaussian
      y_1 = norm.pdf(i, float(mean2[1][0]), np.sqrt(float(covs2[1][0][0])))*weights2[1] # 2nd gaussian
      p_new = y_0 + y_1
      if p > p_new:
          p = p_new
          x_val = i
      else:
          continue
  print("middle point:")
  print(x_val)
  print("probability of the middle point:")
  print(p)

  if verbose > 0:
    # calculate range
    if lownorm == 'lrc':
      minlow = -2.5
      maxlow = 2.5
    else:
      minlow = np.min(lowdf) - 0.5
      maxlow = np.max(lowdf) + 0.5

    # create necessary things to plot
    x_axis = np.arange(minlow, maxlow, pas)
    y_axis0 = norm.pdf(x_axis, float(mean2[0][0]), np.sqrt(float(covs2[0][0][0])))*weights2[0] # 1st gaussian
    y_axis1 = norm.pdf(x_axis, float(mean2[1][0]), np.sqrt(float(covs2[1][0][0])))*weights2[1] # 2nd gaussian

    # Plot histogram with GMM
    plt.hist(x, density=True, color='black', bins=30)
    plt.plot(x_axis, y_axis0, lw=3, c='C0')
    plt.plot(x_axis, y_axis1, lw=3, c='C1')
    plt.plot(x_axis, y_axis0+y_axis1, lw=3, c='C2', ls='dashed')
    plt.xlim(minlow, maxlow)
    plt.xlabel(r"LRC", fontsize=20)
    plt.ylabel(r"Density", fontsize=20)

    # Draw alpha in the plot
    plt.vlines(x=x_val, ymin=0, ymax=p, colors='red', ls='solid', lw=2, label='alpha')
    plt.show()
    plt.savefig('./football_gausmix_2compo.pdf', bbox_inches='tight')
    plt.clf()

  return x_val

# %%
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

def evaluate_communities(groundtruth, communities):
    """
    Evaluate the performance of a set of communities against ground truth.

    Parameters:
    - groundtruth (list of sets): Ground truth communities.
    - communities (list of sets): Detected communities.

    Returns:
    - dict: A dictionary containing evaluation metrics.
    """
    true_labels = [None] * sum([len(comm) for comm in groundtruth])
    pred_labels = [None] * sum([len(comm) for comm in communities])
    for i, gt_community in enumerate(groundtruth):
        for node in gt_community:
            true_labels[node] = i
    for i, community in enumerate(communities):
        for node in community:
            pred_labels[node] = i

    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return {'AMI': ami, 'ARI': ari, 'NMI': nmi}




def generate_LRC(G=None, remove=False, NORM = 'bfc', ATTR = 'value', ALPHA = 10, PAS = 1e-4, MAX_ITER = 10 * 6):
        ### READ IN NETWORK FILE ###

    if NORM == 'lrc':
      G = lowercurv(G)
    else:
      G = balancedcurv(G, alpha=ALPHA)
    df = nx.to_pandas_edgelist(G)
    grouped_by_category = group_nodes_by_feature(G, ATTR)
    groundtruth = []
    for category, nodes in grouped_by_category.items():
        groundtruth.append(nodes)
    gt_set = [set(inner_list) for inner_list in groundtruth]

    ### CALCULATE CURVATURES ###

    G = lowercurv(G)


    edgedf = nx.to_pandas_edgelist(G)
    lowdf = edgedf[NORM]

    ### HISTOGRAM & GMM & FIND ALPHA ###

    beta = estimate_thershold(lowdf, pas=PAS, max_iter=MAX_ITER, lownorm=NORM)

    ## Without removal ##

    if remove:
        remove_curv(beta, G, NORM)
    return G

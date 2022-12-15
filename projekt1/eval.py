#! /usr/bin/python
import scanpy as sc
import anndata
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mstats
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load the h5ad file
def load_data(filename):
    adata = sc.read_h5ad(filename)
    return adata

test_data = load_data("SAD2022Z_Project1_GEX_test.h5ad")
train_data = load_data("SAD2022Z_Project1_GEX_train.h5ad")


# print("Train set has", train_data.obs.shape[0],"observations and", train_data.var.shape[0], "variables.")
# print("Test set has", test_data.obs.shape[0],"observations and", test_data.var.shape[0], "variables.")

# train_raw_d = train_data.layers["counts"].toarray()
#train_raw_d = train_raw_d.reshape([train_raw_d.shape[0]*train_raw_d.shape[1], 1])


#train_raw_nozero = train_raw_d[train_raw_d != 0]

#test_raw_d = test_data.layers["counts"].toarray()
#test_raw_d = test_raw_d.reshape([test_raw_d.shape[0]*test_raw_d.shape[1], 1])
#test_raw_nozero = test_raw_d[test_raw_d != 0]

# train_pp_d = train_data.X.toarray()
# train_pp_d = train_pp_d.reshape([train_pp_d.shape[0]*train_pp_d.shape[1], 1])
#train_pp_nozero = train_pp_d[train_pp_d != 0]
# train_pp_log1p = torch.log1p(torch.Tensor(train_pp_d))
# train_pp_log1p = train_pp_log1p.reshape(train_pp_log1p.shape[0]*train_pp_log1p.shape[1], 1).numpy()
#test_pp_d = test_data.X.toarray()
#test_pp_d = test_pp_d.reshape([test_pp_d.shape[0]*test_pp_d.shape[1], 1])
#test_pp_nozero = test_pp_d[test_pp_d != 0]


def plot_hist(data, bins, range, title, filename):
    plt.hist(data, bins=bins, range=range)
    plt.title(title)
    plt.xlabel("gene expression")
    plt.ylabel("count")
    plt.savefig(filename)
    plt.close()

# plot_hist(train_raw_d, 200, [0,10], "Gene expression distribution in all cells, range: 0 to 10, train raw data.",\
#     "train_raw_0_10.png")

# plot_hist(train_raw_d, 200, [0.5,10], "Gene expression distribution in all cells, range: 0.5 to 10, train raw data.",\
#     "train_raw_05_10.png")

# plot_hist(train_raw_d, 100, None, "Gene expression distribution in all cells, full range, train raw data.",\
#     "train_raw_full.png")

# plot_hist(test_raw_d, 100, None, "Gene expression distribution in all cells, full range, test raw data.",\
#     "test_raw_full.png")

# plot_hist(test_raw_d, 200, [0, 10], "Gene expression distribution in all cells, range: 0 to 10, test raw data.",\
#     "test_raw_0_10.png")

# plot_hist(test_raw_d, 200, [0.5, 10], "Gene expression distribution in all cells, range: 0.5 to 10, test raw data.",\
#     "test_raw_05_10.png")


# plot_hist(train_pp_d, 200, [0,10], "Gene expression distribution in all cells, range: 0 to 10, train preprocessed data.",\
#     "train_pp_0_10.png")

# plot_hist(train_pp_d, 200, [0.5,10], "Gene expression distribution in all cells, range: 0.5 to 10, train preprocessed data.",\
#     "train_pp_05_10.png")

# plot_hist(train_pp_d, 100, None, "Gene expression distribution in all cells, full range, train preprocessed data.",\
#     "train_pp_full.png")

# plot_hist(test_pp_d, 100, None, "Gene expression distribution in all cells, full range, test preprocessed data.",\
#     "test_pp_full.png")

# plot_hist(test_pp_d, 200, [0, 10], "Gene expression distribution in all cells, range: 0 to 10, test preprocessed data.",\
#     "test_pp_0_10.png")

# plot_hist(test_pp_d, 200, [0.5, 10], "Gene expression distribution in all cells, range: 0.5 to 10, test preprocessed data.",\
#     "test_pp_05_10.png")


# plot_hist(train_pp_nozero, 50, None, "Gene expression dist. in all cells, full range, train preprocessed data w/o zeros.",\
#     "train_pp_full_nozeros.png")

# plot_hist(train_pp_nozero, 100, [0,20], "Gene expression dist. in all cells, 0 to 20, train preprocessed data w/o zeros.",\
#     "train_pp_nozeros_0_20.png")

# plot_hist(test_pp_nozero, 50, None, "Gene expression dist. in all cells, full range, test preprocessed data w/o zeros.",\
#     "test_pp_full_nozeros.png")

# plot_hist(test_pp_nozero, 100, [0,20], "Gene expression dist. in all cells, 0 to 20, test preprocessed data w/o zeros.",\
#     "test_pp_nozeros_0_20.png")

# plot_hist(train_raw_nozero, 50, None, "Gene expression dist. in all cells, full range, train raw data w/o zeros.",\
#     "train_raw_full_nozeros.png")

# plot_hist(train_raw_nozero, 100, [0,20], "Gene expression dist. in all cells, 0 to 20, train raw data w/o zeros.",\
#     "train_raw_nozeros_0_20.png")

# plot_hist(test_raw_nozero, 50, None, "Gene expression dist. in all cells, full range, test raw data w/o zeros.",\
#     "test_raw_full_nozeros.png")

# plot_hist(test_raw_nozero, 100, [0,20], "Gene expression dist. in all cells, 0 to 20, test raw data w/o zeros.",\
#     "test_raw_nozeros_0_20.png")

# plot_hist(train_pp_log1p, 100, None, "Gene expression dist. in all cells, full range, train pp. data log1p.",\
#    "train_pp_full_log.png")

# plot_hist(train_pp_log1p, 200, [0.5,10], "Gene expression dist. in all cells, 0.5 to 10, train pp. data log1p.",\
#    "train_pp_05_10_log.png")

def stats(array):
    print("Minimal value:", array.min())
    print("Maximal value:", array.max())
    print("Mean:", np.mean(array))
    print("Median:", np.median(array))
    print("There are", (array == 0).sum(), "zeros.")
    print("There are", (array == 1).sum(), "ones.")
    print("There are", (array>10).sum(), "values greater than 10.")

def obs_info(data):
    print(data.obs.columns)
    print(set(data.obs["Site"]))

def plot_loss(filename, title, outname):
  with open(filename, "r") as loss:
    loss = loss.read().strip().split("\n")
    loss = [float(l) for l in loss]
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    plt.plot(range(len(loss)), loss)
    plt.title(title)
    plt.xlabel("No. iteration")
    plt.ylabel("-ELBO")
    plt.savefig(outname, pad_inches=0.5)
    plt.close()

def latent_space_PCA(filename):
  df = pd.read_csv(filename, header = None)
  #print(df)
  x = df.values
  x = StandardScaler().fit_transform(x)
  pca = PCA(.95)
  pca.fit(x)
  print(pca.n_components_)

plot_loss("TRAIN_ELBO_b1_epochs1_hdim200_zdim50_lr0.0005_batch20CUSTOM.txt", \
          "-ELBO, training, latent dim 50, neg. binomial decoder","TRAIN_ELBO_b1_epochs1_hdim200_zdim50_lr0.0005_batch20_negbin.png")

plot_loss("TRAIN_REC_LOSS_b1_epochs1_hdim200_zdim50_lr0.0005_batch20CUSTOM.txt",\
          "Reconstruction loss, training, latent dim 50, neg. binomial decoder", "TRAIN_REC_LOSS_b1_epochs1_hdim200_zdim50_lr0.0005_batch20_negbin.png")

plot_loss("TRAIN_REG_LOSS_b1_epochs1_hdim200_zdim50_lr0.0005_batch20CUSTOM.txt",\
          "Regularization loss, training, latent dim 50, neg. binomial decoder", "TRAIN_REG_LOSS_b1_epochs1_hdim200_zdim50_lr0.0005_batch20_negbin.png")

plot_loss("TEST_ELBO_b1_epochs1_hdim200_zdim50_lr0.0005_batch20CUSTOM.txt", \
          "-ELBO, training, test set, latent dim 50, neg. binomial decoder","TEST_ELBO_b1_epochs1_hdim200_zdim50_lr0.0005_batch20_negbin.png")

plot_loss("TEST_REC_LOSS_b1_epochs1_hdim200_zdim50_lr0.0005_batch20CUSTOM.txt",\
          "Reconstruction loss, training, test set, latent dim 50, neg. binomial decoder", "TEST_REC_LOSS_b1_epochs1_hdim200_zdim59_lr0.0005_batch20_negbin.png")

plot_loss("TEST_REG_LOSS_b1_epochs1_hdim200_zdim50_lr0.0005_batch20CUSTOM.txt",\
          "Regularization loss, training, test set, latent dim 50, neg. binomial decoder", "TEST_REG_LOSS_b1_epochs1_hdim200_zdim50_lr0.0005_batch20_negbin.png")

#! /usr/bin/python

import torch
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import scanpy as sc
import csv
import copy
import re
import sys
import numpy as np
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from models import Network, EncoderGaussian, DecoderGaussian, VAE, RNAseq_Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fract = 0.005


def name_file(prefix, beta, num_epochs, h_dim, z_dim, lr, batch_size, format):
  # produces file names based on model parameters
  return prefix + "_b"+str(beta)+"_epochs"+str(num_epochs)+"_hdim"+str(h_dim)+"_zdim"+str(z_dim)+"_lr"+str(lr)+\
    "_batch"+str(batch_size)+format

def load_data(filename):
  # data loading
    adata = sc.read_h5ad(filename)
    return adata


test_data = load_data("SAD2022Z_Project1_GEX_test.h5ad")
# taking only fraction if necessary
test_gex = sc.pp.subsample(test_data, fraction = fract, copy = True, random_state = 5)
# log1p application and conversion to dense array
test_dataset = torch.log1p(torch.Tensor(test_gex.X.toarray())).to(DEVICE)
# analogically for training dataset
train_data = load_data("SAD2022Z_Project1_GEX_train.h5ad")
train_gex = sc.pp.subsample(train_data, fraction = fract, copy = True, random_state = 5)
train_dataset = torch.log1p(torch.Tensor(train_gex.X.toarray())).to(DEVICE)
# saving vector with cell types
train_cts = train_gex.obs["cell_type"]
test_cts = test_gex.obs["cell_type"]

def train(BETA, NUM_EPOCHS, INPUT_DIM, H_DIM, Z_DIM, LR, BATCH_SIZE):
  # producing files for saving: loss with division on -elbo, reconstruction loss and regularization loss,
  # file with latent space for last epoch and file with model parameters
  lspace = open(name_file("TRAIN_LATENTSPACE", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".csv"), "w+")
  latent_writer = csv.writer(lspace, delimiter=",")
  train_elbo = open(name_file("TRAIN_ELBO", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".txt"), "w+")
  train_reconst_loss = open(name_file("TRAIN_REC_LOSS", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".txt"), "w+")
  train_regul_loss = open(name_file("TRAIN_REG_LOSS", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".txt"), "w+")
  test_elbo = open(name_file("TEST_ELBO", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".txt"), "w+")
  test_reconst_loss = open(name_file("TEST_REC_LOSS", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".txt"), "w+")
  test_regul_loss = open(name_file("TEST_REG_LOSS", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".txt"), "w+")
  model_file = open(name_file("MODEL", BETA, NUM_EPOCHS, H_DIM, Z_DIM, LR, BATCH_SIZE, ".txt"), "w+")

  for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, y) in loop:
      x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
      z, x_hat = model.forward(x)
      # compute loss
      def loss_fn(x, beta):
        reconstruction_loss = model.decoder.log_prob(x)
        regularization_loss = beta*torch.sum(torch.distributions.kl.kl_divergence(model.prior_dist, model.encoder.dist))
        elbo = reconstruction_loss - regularization_loss
        return -elbo, reconstruction_loss, regularization_loss

      test_sc, test_sct = test_iter.next()
      te_elbo, te_rec_loss, te_reg_loss = loss_fn(BETA, test_sc)
      optimazer.zero_grad()
      tr_elbo, tr_rec_loss, tr_reg_loss = loss_fn(BETA, x)
      tr_elbo.backward()
      optimazer.step()

      loop.set_postfix(loss = tr_elbo.item())
      train_elbo.write(str(tr_elbo.item())+"\n")
      train_reconst_loss.write(str(tr_rec_loss.item())+"\n")
      train_regul_loss.write(str(tr_reg_loss.item())+"\n")
      test_elbo.write(str(te_elbo.item())+"\n")
      test_reconst_loss.write(str(te_rec_loss.item())+"\n")
      test_regul_loss.write(str(te_reg_loss.item())+"\n")

      if epoch == NUM_EPOCHS - 1:
        z = z.cpu().detach().numpy()
        for _, x in enumerate(z):
          latent_writer.writerow(list(x))

        
  model_file.write(">Encoder distribution\n" +">"+ str(model.encoder.mu.cpu().detach().numpy()) + "\n>"\
    + str(model.encoder.var.cpu().detach().numpy()) + "\n" +\
    ">Decoder distribution\n>" + str(model.decoder.mu.cpu().detach().numpy()) +\
      "\n>" + str(model.decoder.var.cpu().detach().numpy()))



train_dataset = RNAseq_Dataset(train_dataset, train_cts)
test_dataset = RNAseq_Dataset(test_dataset, test_cts)
train_loader = DataLoader(dataset = train_dataset, batch_size = 10, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True)
test_iter = iter(test_loader)
net = Network(5000, 300, 100).to(DEVICE)
encoder = EncoderGaussian(net).to(DEVICE)
decoder = DecoderGaussian(net).to(DEVICE)
model = VAE(encoder, decoder).to(DEVICE)
optimazer = optim.Adam(model.parameters(), lr = 0.005)

def convert_params(List):
  res = []
  for num in List:
    try:
      num = float(num)
      res.append(num)
    except:
      num = re.findall("\d+\.*\d*", num)
      if len(num) != 0:
        res.append(float(num[0]))
  return res

def load_model(filename, INPUT_DIM, H_DIM, Z_DIM):
  data = open(filename).read().split(">")
  net = Network(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
  encoder_ = EncoderGaussian(net).to(DEVICE)
  decoder_ = DecoderGaussian(net).to(DEVICE)

  enc_mu = torch.Tensor(convert_params(re.split("\s+", data[2])))
  enc_var = torch.Tensor(convert_params(re.split("\s+", data[3])))
  enc_cov = torch.diag_embed(enc_var)
  encoder_.mu = enc_mu
  encoder_.dist = torch.distributions.MultivariateNormal(enc_mu, enc_cov)

  dec_mu = torch.Tensor(convert_params(re.split("\s+", data[5])))
  dec_var = torch.Tensor(convert_params(re.split("\s+", data[6])))
  dec_cov = torch.diag_embed(dec_var)
  decoder_.dist = torch.distributions.MultivariateNormal(dec_mu, dec_cov)
  
  return VAE(encoder_, decoder_)

def test(M, dataloader, INPUT_DIM):
  Z = []
  labels = []
  M = M.to(DEVICE)
  elbo = []
  rec_loss = []
  reg_loss = []
  loop = tqdm(enumerate(dataloader))
  with torch.no_grad():
    for i, (x, y) in loop:
      # Forward pass
      try:
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        z, _ = M.forward(x)
        Z.append(z.numpy()[0])
        el, rec, reg = M.loss_fn(x)
        elbo.append(el)
        rec_loss.append(rec)
        reg_loss.append(reg)
        labels.append(y)
      except RuntimeError:
        print("Runtime Error")
        break
  labels = [l[0] for l in labels]
  print("-ELBO after last iteration of testing on model with latent space size", M.encoder.mu.shape, elbo[-1])
  return Z, labels


def pca_encoded_cells(df, plot_title):
  x = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values
  x = StandardScaler().fit_transform(x)
  targets = list(set(y))
  pca = PCA(n_components=2)
  principalComponents = pca.fit_transform(x)
  principalDf = pd.DataFrame(data = principalComponents\
             , columns = ['principal component 1', 'principal component 2'])
  finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
  fig = plt.figure(figsize = (14,8))
  ax = fig.add_subplot(1,1,1) 
  ax.set_xlabel('Principal Component 1', fontsize = 15)
  ax.set_ylabel('Principal Component 2', fontsize = 15)
  ax.set_title('2 component PCA', fontsize = 20)
  for target in targets:
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], \
      finalDf.loc[indicesToKeep, 'principal component 2'], s = 50)
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
  plt.savefig(plot_title)
  plt.close()


model_z50 = load_model("E1_ZDIM_50/MODEL_b1_epochs1_hdim300_zdim50_lr0.005_batch10.txt", 5000, 300, 50)
Z_50, labels_50 = test(model_z50, test_loader, 5000)
df_50 = pd.DataFrame(Z_50)
df_50['target'] = labels_50
pca_encoded_cells(df_50, "PCA_test_zdim50.png")


model_z100 = load_model("E1_ZDIM_100/MODEL_b1_epochs1_hdim300_zdim100_lr0.005_batch10.txt", 5000, 300, 100)
Z_100, labels_100 = test(model_z100, test_loader, 5000)
df_100 = pd.DataFrame(Z_100)
df_100['target'] = labels_100
pca_encoded_cells(df_100, "PCA_test_zdim100.png")


model_z150 = load_model("E1_ZDIM_150/MODEL_b1_epochs1_hdim300_zdim150_lr0.005_batch10.txt", 5000, 300, 100)
Z_150, labels_150 = test(model_z150, test_loader, 5000)
df_150 = pd.DataFrame(Z_150)
df_150['target'] = labels_150
pca_encoded_cells(df_150, "PCA_test_zdim150.png")

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

class Network(torch.nn.Module):
  def __init__(self, input_dim, h_dim = 200, z_dim=20, o_dim=20):
    super(Network, self).__init__()
    self.img2hid = nn.Linear(input_dim, h_dim)
    self.hid2mu_e = nn.Linear(h_dim, z_dim)
    self.hid2sigma_e = nn.Linear(h_dim, z_dim)

    self.z2hid = nn.Linear(z_dim, h_dim)
    self.hid2mu_d = nn.Linear(h_dim, input_dim)
    self.hid2sigma_d = nn.Linear(h_dim, input_dim)

    self.relu = nn.ReLU()

  def encode(self, x):
    h = self.relu(self.img2hid(x))
    mu, sigma = self.hid2mu_e(h), self.hid2sigma_e(h)
    return mu, sigma

  def decode(self, z):
    h = self.relu(self.z2hid(z))
    mu, sigma = self.hid2mu_d(h), self.hid2sigma_d(h)
    return mu, sigma

class EncoderGaussian(torch.nn.Module):
  def __init__(self, network):
    super(EncoderGaussian, self).__init__()
    self.network = network
    self.dist = None
    self.mu = None
    self.var = None
    self.log_var = None

  def log_prob(self,x):
    return self.dist.log_prob(x).sum(-1)

  def _sample(self):
    return self.dist.rsample()
  
  def forward(self,x):
    mu, log_var = self.network.encode(x)
    var = torch.exp(log_var)
    cov = torch.diag_embed(var)
    self.dist = torch.distributions.MultivariateNormal(mu, cov)
    self.mu = mu
    self.log_var = log_var
    self.var = var
    return mu, var

class DecoderGaussian(torch.nn.Module):
  def __init__(self, network):
    super(DecoderGaussian, self).__init__()
    self.network = network
    self.mu = None
    self.var = None
    self.dist = None

  def _sample(self):
    return torch.sigmoid(self.dist.sample()) #??sigmoud??

  def log_prob(self,x):
    return self.dist.log_prob(x).sum(-1)

  def forward(self,x):
    mu, log_var = self.network.decode(x)
    var = torch.exp(log_var)
    cov = torch.diag_embed(var)
    self.dist = torch.distributions.MultivariateNormal(mu, cov)
    self.mu = mu
    self.var = var
    return mu, var

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.prior_dist = torch.distributions.MultivariateNormal(torch.zeros([Z_DIM]).to(DEVICE), torch.eye(Z_DIM).to(DEVICE))

  def loss_fn(self, x):
    reconstruction_loss = self.decoder.log_prob(x)
    regularization_loss = BETA*torch.sum(torch.distributions.kl.kl_divergence(self.prior_dist, self.encoder.dist))
    elbo = reconstruction_loss - regularization_loss
    return -elbo, reconstruction_loss, regularization_loss

  def encode(self, x):
    self.encoder.forward(x)
    z = self.encoder._sample()
    return z

  def decode(self, z):
    self.decoder.forward(z)
    x_hat = self.decoder._sample()
    return x_hat
    
  def forward(self,x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return z, x_hat

class RNAseq_Dataset(Dataset):
  def __init__(self, matrix, cell_types):
    self.rows = [matrix[i, :] for i in range(matrix.shape[0])]
    self.targets = list(zip(self.rows, cell_types))

  def __getitem__(self, index):
    sc, ct = self.targets[index]
    return sc, ct
    #return torch.Tensor(self.matrix[index, :])

  def __len__(self):
      return len(self.targets)

def name_file(prefix, beta, num_epochs, h_dim, z_dim, lr, batch_size, format):
  return prefix + "_b"+str(beta)+"_epochs"+str(num_epochs)+"_hdim"+str(h_dim)+"_zdim"+str(z_dim)+"_lr"+str(lr)+\
    "_batch"+str(batch_size)+format

def load_data(filename):
    adata = sc.read_h5ad(filename)
    return adata

INPUT_DIM = 5000
H_DIM = 300
Z_DIM = 100
NUM_EPOCHS = 1
BATCH_SIZE = 10
LR = 0.005
BETA = 1


fract = 0.005
test_data = load_data("SAD2022Z_Project1_GEX_test.h5ad")
test_gex = sc.pp.subsample(test_data, fraction = fract, copy = True, random_state = 5)
test_dataset = torch.log1p(torch.Tensor(test_gex.X.toarray())).to(DEVICE)

train_data = load_data("SAD2022Z_Project1_GEX_train.h5ad")
train_gex = sc.pp.subsample(train_data, fraction = fract, copy = True, random_state = 5)
train_dataset = torch.log1p(torch.Tensor(train_gex.X.toarray())).to(DEVICE)

train_cts = train_gex.obs["cell_type"]
test_cts = test_gex.obs["cell_type"]

def train():
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
      # Forward pass
      print(x.shape)
      x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
      z, x_hat = model.forward(x)
      print("Z shape", z.shape)
      # compute loss
      def loss_fn(x):
        reconstruction_loss = model.decoder.log_prob(x)
        regularization_loss = BETA*torch.sum(torch.distributions.kl.kl_divergence(model.prior_dist, model.encoder.dist))
        elbo = reconstruction_loss - regularization_loss
        return -elbo, reconstruction_loss, regularization_loss

      test_sc, test_sct = test_iter.next()
      te_elbo, te_rec_loss, te_reg_loss = loss_fn(test_sc)
      optimazer.zero_grad()
      tr_elbo, tr_rec_loss, tr_reg_loss = loss_fn(x)
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
train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True)
test_iter = iter(test_loader)
net = Network(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
encoder = EncoderGaussian(net).to(DEVICE)
decoder = DecoderGaussian(net).to(DEVICE)
model = VAE(encoder, decoder).to(DEVICE)
optimazer = optim.Adam(model.parameters(), lr = LR)

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

def load_model(filename):
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

def test(M, dataloader):
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

        #loop.set_postfix(loss = el.item())
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

Z_DIM = 50
model_z50 = load_model("E1_ZDIM_50/MODEL_b1_epochs1_hdim300_zdim50_lr0.005_batch10.txt")
Z_50, labels_50 = test(model_z50, test_loader)
df_50 = pd.DataFrame(Z_50)
df_50['target'] = labels_50
pca_encoded_cells(df_50, "PCA_test_zdim50.png")

Z_DIM = 100
model_z100 = load_model("E1_ZDIM_100/MODEL_b1_epochs1_hdim300_zdim100_lr0.005_batch10.txt")
Z_100, labels_100 = test(model_z100, test_loader)
df_100 = pd.DataFrame(Z_100)
df_100['target'] = labels_100
pca_encoded_cells(df_100, "PCA_test_zdim100.png")

Z_DIM = 150
model_z150 = load_model("E1_ZDIM_150/MODEL_b1_epochs1_hdim300_zdim150_lr0.005_batch10.txt")
Z_150, labels_150 = test(model_z150, test_loader)
df_150 = pd.DataFrame(Z_150)
df_150['target'] = labels_150
pca_encoded_cells(df_150, "PCA_test_zdim150.png")

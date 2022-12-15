#! /usr/bin/python

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F

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
    self.prior_dist = torch.distributions.MultivariateNormal(torch.zeros([self.encoder.mu.shape[1]]).to(DEVICE), torch.eye(self.encoder.mu.shape[1]).to(DEVICE))

  def loss_fn(self, x, beta):
    reconstruction_loss = self.decoder.log_prob(x)
    regularization_loss = beta*torch.sum(torch.distributions.kl.kl_divergence(self.prior_dist, self.encoder.dist))
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

class NetworkCustom(torch.nn.Module):
  def __init__(self, input_dim, h_dim = 200, z_dim=20, o_dim=20):
    super(NetworkCustom, self).__init__()
    self.img2hid = nn.Linear(input_dim, h_dim)
    self.hid2mu_e = nn.Linear(h_dim, z_dim)
    self.hid2sigma_e = nn.Linear(h_dim, z_dim)

    self.z2hid = nn.Linear(z_dim, h_dim)
    self.hid2tot_count = nn.Linear(h_dim, input_dim)
    self.hid2probs = nn.Linear(h_dim, input_dim)
    self.hid2logits = nn.Linear(h_dim, input_dim)

    self.relu = nn.ReLU()

  def encode(self, x):
    h = self.relu(self.img2hid(x))
    mu, sigma = self.hid2mu_e(h), self.hid2sigma_e(h)
    return mu, sigma

  def decode(self, z):
    h = self.relu(self.z2hid(z))
    tc, probs, logits = self.hid2tot_count(h), self.hid2probs(h), self.hid2logits(h)
    return tc, probs, logits

class DecoderNegBinomial(torch.nn.Module):
  def __init__(self, network):
    super(DecoderNegBinomial, self).__init__()
    self.network = network
    self.dist = None

  def _sample(self):
    self.dist.sample()

  def log_prob(self,x):
    return self.dist.log_prob(x).sum(-1)

  def forward(self,x):
    tc, probs, logits = self.network.decode(x)
    self.dist = torch.distributions.NegativeBinomial(tc, probs, logits)
    self.tc = tc
    self.probs = probs
    self.logits = logits
    return tc, probs, logits
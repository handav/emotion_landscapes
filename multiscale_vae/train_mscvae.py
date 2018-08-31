import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
import itertools
import progressbar
import numpy as np
import sys
sys.path.append('../')
print('Cuda is available:', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/mscvae/', help='base directory to save logs')
parser.add_argument('--data_root', default='data/', help='base directory to save logs')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=32, help='the height / width of the input image to network: 32 | 64')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='emotion_landscapes', help='dataset to train with: emotion_landscapes | cifar')
parser.add_argument('--z_dim', default=128, type=int, help='dimensionality of latent space')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--nclass', type=int, default=9, help='number of classes (should be 9 for emotion landscapes)')
parser.add_argument('--save_model', action='store_true', help='if true, save the model throughout training')
parser.add_argument('--beta', default=0.001, type=float, help='learning rate')
parser.add_argument('--all_labels', action='store_true', help='if true, give full distribution over labels to G and D')
parser.add_argument('--nlevels', type=int, default=3, help='number of levels')

opt = parser.parse_args()

name = 'z_dim=%d-lr=%.5f-beta=%.4f-all_labels=%s' % (opt.z_dim, opt.lr, opt.beta, opt.all_labels)
opt.log_dir = '%s/%s_%dx%d/%s' % (opt.log_dir, opt.dataset, opt.image_width, opt.image_width, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/rec/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

# ---------------- load the models ----------------
if opt.image_width == 32:
    import models.msvae_32x32 as models
elif opt.image_width == 64:
    import models.msvae_64x64 as models
netE = [models.conditional_stochastic_encoder(opt.z_dim, opt.nclass, opt.channels*(2 if i > 0 else 1)) for i in range(opt.nlevels)]
netD = []
for i in range(opt.nlevels):
    if i == 0:
        netD.append(models.decoder(opt.z_dim+opt.nclass, opt.channels))
    else:
        netD.append(models.img_conditional_decoder(opt.z_dim+opt.nclass, opt.channels))

netE_optimizer = [opt.optimizer(netE[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for i in range(opt.nlevels)]
netD_optimizer = [opt.optimizer(netD[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for i in range(opt.nlevels)]
for i in range(opt.nlevels):
    netE[i].apply(utils.init_weights)
    netD[i].apply(utils.init_weights)

    netE[i].cuda()
    netD[i].cuda()

mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
mse_criterion.cuda()
l1_criterion.cuda()

def multiscale(x):
    scales = []
    tmp = []
    for i in range(opt.nlevels):
        if i == 0:
            scales.append(x)
            tmp.append(x)
        else:
            tmp.append(nn.AvgPool2d(2, 2)(tmp[i-1]))
            scales.append(nn.Upsample(scale_factor=2**i)(tmp[i]))
    return [scales[i] for i in range(opt.nlevels-1, -1, -1)]

def fixed_kl_criterion(mu, logvar):
  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  # Normalise by same number of elements as in reconstruction
  KLD /= opt.batch_size  
  return KLD

# def kl_criterion(mu1, logvar1, mu2, logvar2):
#     # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
#     #   log( sqrt(
#     # 
#     sigma1 = logvar1.mul(0.5).exp() 
#     sigma2 = logvar2.mul(0.5).exp() 
#     kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
#     return kld.sum() / opt.batch_size
# ---------------- datasets ----------------

# loads the dataset
trainset = utils.load_dataset(opt) 
train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=opt.data_threads)


def get_training_batch():
    while True:
        if opt.all_labels:
            for x, y, yp in train_loader:
                yield [x.cuda(), yp.cuda()]
        else:
            for x, y in train_loader:
                y1h = torch.zeros(opt.batch_size, opt.nclass).cuda()
                y1h = y1h.scatter_(1, y.data.view(opt.batch_size, 1), 1)
                yield [x.cuda(), y1h.cuda()]
training_batch_generator = get_training_batch()

# same z for each column
def make_plot_z():
    z = torch.randn(opt.batch_size, opt.z_dim, 1, 1).cuda()
    nrow = opt.nclass 
    ncol = int(opt.batch_size/nrow) 
    for j in range(ncol):
        zz = torch.randn(opt.z_dim, 1, 1)
        for i in range(nrow):
            z[i*ncol+j].copy_(zz)
    return z

# create the reconstructed images
def plot_rec(x, y, s, epoch):
    y = y.view(opt.batch_size, opt.nclass, 1, 1)

    if s == 0:
        _, z, _ = netE[s]((x[s], y))
        rec = netD[s](torch.cat([z, y], 1))
    else:
        residual = x[s] - x[s-1]
        _, z, _ = netE[s]((torch.cat([x[s], residual], 1), y))
        rec = netD[s]([x[s-1], torch.cat([z, y], 1)])

    to_plot = []
    nrow = opt.nclass 
    ncol = int(opt.batch_size/nrow)
    for i in range(nrow):
        row = []
        for j in range(ncol):
            row.append(x[s][i*ncol+j])
            #if s > 0: 
            #    row.append(residual[i*ncol+j])
            row.append(rec[i*ncol+j])
        to_plot.append(row)

    fname = '%s/rec/%d_%d.png' % (opt.log_dir, epoch, opt.image_width/(2**(opt.nlevels-s-1))) 
    utils.save_tensors_image(fname, to_plot)

# this will plot such that each row is a different class and each column is a different z
def plot_gen(epoch):
    nrow = opt.nclass 
    ncol = int(opt.batch_size/nrow) 
    y = torch.Tensor(opt.batch_size, opt.nclass, 1, 1).cuda().zero_()
    for i in range(nrow):
        for j in range(ncol):
            y[i*ncol+j][i] = 1
    scales = []
    residuals = []
    for s in range(opt.nlevels):
        if s == 0:
            z = make_plot_z() #torch.cuda.FloatTensor(opt.batch_size, opt.z_dim, 1, 1).normal_()
            gen = netD[s](torch.cat([z, y], 1)).detach()
            residual = 0
        else:
            z = make_plot_z() #torch.cuda.FloatTensor(opt.batch_size, opt.z_dim, 1, 1).normal_()
            gen = netD[s]([scales[s-1], torch.cat([z, y], 1)]).detach()
            #gen = nn.Sigmoid()(scales[s-1] + residual)
            #residual.data = residual.data.mul(0.5).add(0.5)
        scales.append(gen)
        #residuals.append(residual)
        
    to_plot = []
    for i in range(nrow):
        row = []
        for j in range(ncol):
            row.append(scales[-1][i*ncol+j])
        to_plot.append(row)

    fname = '%s/gen/%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

    to_plot = []
    nrow = 6 
    ncol = 3 
    for i in range(nrow):
        row = []
        for j in range(ncol):
            row.append(scales[0][i*ncol+j])
            for s in range(1, opt.nlevels):
                #row.append(residuals[s][i*ncol+j])
                row.append(scales[s][i*ncol+j])
        to_plot.append(row)

    fname = '%s/gen/scales_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

# train with x, y (labels), s (level number)
def train(x, y, s):
    y = y.view(opt.batch_size, opt.nclass, 1, 1)
    netE[s].zero_grad()
    netD[s].zero_grad()
    #z, mu, logvar = netE[i](x - canvas[i])
    # for the first level, 
    if s == 0:
        z, mu, logvar = netE[s]((x[s], y))
        rec = netD[s](torch.cat([z, y], 1))
    else:
        residual = x[s] - x[s-1]
        z, mu, logvar = netE[s]((torch.cat([x[s], residual], 1), y))
        rec = netD[s]([x[s-1], torch.cat([z, y], 1)])

    # get loss: mean squared error (mse) and kl-divergence (kld)
    mse = l1_criterion(rec, x[s])
    kld = fixed_kl_criterion(mu, logvar)

    if s == 0:
        loss = mse + opt.beta*kld
    else:
        loss = mse + 0.1*opt.beta*kld
    loss.backward()

    netE_optimizer[s].step()
    netD_optimizer[s].step()

    return mse.item(), kld.item()

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    for s in range(opt.nlevels):
        netE[s].train()
        netD[s].train()
    epoch_mse = [0 for i in range(opt.nlevels)]
    epoch_kld = [0 for i in range(opt.nlevels)]
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x, y = next(training_batch_generator)
        ms_x = multiscale(x)

        for s in range(opt.nlevels):
            mse, kld = train(ms_x, y, s)
            epoch_mse[s] += mse
            epoch_kld[s] += kld

    progress.finish()
    utils.clear_progressbar()

    print('Epoch %02d : %d examples' % (epoch, epoch*opt.epoch_size))
    for s in range(opt.nlevels): 
        print('mse loss: %.5f | kld loss: %.5f' % (epoch_mse[s]/opt.epoch_size, epoch_kld[s]/opt.epoch_size))

    # plot some stuff
    for s in range(opt.nlevels):
        netE[s].eval()
        netD[s].eval()
    x, y = next(training_batch_generator)
    ms_x = multiscale(x)
    for s in range(opt.nlevels):
        plot_rec(ms_x, y, s, epoch)
    plot_gen(epoch)

    # save the model
    if opt.save_model and epoch % 10 == 0:
        torch.save({
            'netD': netD,
            'netE': netE,
            'opt': opt},
            '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
           

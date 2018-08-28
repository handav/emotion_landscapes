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

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/vae/', help='base directory to save logs')
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
# the generator maps [z, y_vec] -> x
# z dimensions: [batch_size, z_dim, 1, 1]
# y_vec dimensions: [batch_size, nclass, 1, 1]
#
# the discriminator maps [x, y_im] -> prediction
# x dimensions: [batch_size, channels, image_width, image_width]
# y_im dimensions: [batch_size, nclass, image_width, image_width] (i.e., one hot vector expanded to be of dimensionality of image)
import models.cvae as models
if opt.image_width == 32:
    model = models.cvae_32x32(opt.z_dim, opt.nclass, opt.channels)
else:
    raise ValueError('Invalid image width %d' % opt.image_width)


model.apply(utils.init_weights)

optimizer = opt.optimizer(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

model.cuda()

# loss function for discriminator
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
mse_criterion.cuda()
l1_criterion.cuda()

# KL_divergence( N(mu, logvar) | N(0, 1) )
def kl_criterion(mu, logvar):
  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  # Normalise by same number of elements as in reconstruction
  KLD /= opt.batch_size  

  return KLD

# ---------------- datasets ----------------
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
                yield [x.cuda(), y.cuda(), yp.cuda()]
        else:
            for x, y in train_loader:
                yield [x.cuda(), y.cuda(), None]
training_batch_generator = get_training_batch()

# so all our generations use same noise vector - useful for visualizaiton purposes
z_fixed = torch.randn(opt.batch_size, opt.z_dim, 1, 1).cuda()
nrow = opt.nclass 
ncol = int(opt.batch_size/nrow) 
for j in range(ncol):
    zz = torch.randn(opt.z_dim, 1, 1)
    for i in range(nrow):
        z_fixed[i*ncol+j].copy_(zz)

def plot_gen(x, epoch):
    x, y, yp = x

    nrow = opt.nclass 
    ncol = int(opt.batch_size/nrow) 

    if opt.all_labels:
        y_onehot = yp.view(opt.batch_size, opt.nclass, 1, 1)

        # different class per row
        y_onehot = torch.Tensor(opt.batch_size, opt.nclass, 1, 1).cuda().zero_()
        for i in range(nrow):
            for j in range(ncol):
                y_onehot[i*ncol+j][i] = 1
        gen = model.decode(z_fixed, y_onehot)

        to_plot = []
        for i in range(nrow):
            row = []
            for j in range(ncol):
                row.append(gen[i*ncol+j])
            to_plot.append(row)

        fname = '%s/gen/p_%d.png' % (opt.log_dir, epoch) 
        utils.save_tensors_image(fname, to_plot)

    y_onehot = y_onehot.view(opt.batch_size, opt.nclass, 1, 1)

    # different class per row
    y_onehot = torch.Tensor(opt.batch_size, opt.nclass, 1, 1).cuda().zero_()
    for i in range(nrow):
        for j in range(ncol):
            y_onehot[i*ncol+j][i] = 1
    gen = model.decode(z_fixed, y_onehot)

    to_plot = []
    for i in range(nrow):
        row = []
        for j in range(ncol):
            row.append(gen[i*ncol+j])
        to_plot.append(row)

    fname = '%s/gen/%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

def plot_rec(x, epoch):
    x, y, yp = x

    # convert the integer y into a one_hot representation for decoder 
    if opt.all_labels:
        y_onehot = yp
    else:
        y_onehot = torch.Tensor(opt.batch_size, opt.nclass).cuda().zero_()
        y_onehot.scatter_(1, y.data.view(opt.batch_size, 1).long(), 1)
    y_onehot = y_onehot.view(opt.batch_size, opt.nclass, 1, 1)

    rec, _, _= model((x, y_onehot))

    to_plot = []
    nrow, ncol = 8, 8
    for i in range(nrow):
        row = []
        for j in range(ncol):
            row.append(x[i*ncol+j])
            row.append(rec[i*ncol+j])
        to_plot.append(row)

    utils.save_tensors_image('%s/rec/%d.png' %(opt.log_dir, epoch), to_plot)

def train(x):
    x, y, yp = x

    # convert the integer y into a one_hot representation for decoder 
    if opt.all_labels:
        y_onehot = yp
    else:
        y_onehot = torch.Tensor(opt.batch_size, opt.nclass).cuda().zero_()
        y_onehot.scatter_(1, y.data.view(opt.batch_size, 1).long(), 1)
    y_onehot = y_onehot.view(opt.batch_size, opt.nclass, 1, 1)

    model.zero_grad()

    rec, mu, logvar = model([x, y_onehot])
    mse = mse_criterion(rec, x)
    kld = kl_criterion(mu, logvar)
    loss = mse + opt.beta*kld

    loss.backward() # computes gradients
    optimizer.step() # updates parameters

    return mse.item(), kld.item()

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    model.train()
    epoch_mse, epoch_kld = 0, 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        mse, kld = train(x)

        epoch_mse += mse
        epoch_kld += kld

    print('[%02d] mse = %.4f | kld = %.4f' % (epoch, epoch_mse / opt.epoch_size, epoch_kld / opt.epoch_size))

    progress.finish()
    utils.clear_progressbar()

    # plot some stuff
    model.eval()

    # XXX: se should have some test data
    x = next(training_batch_generator)
    plot_rec(x, epoch)
    plot_gen(x, epoch)

    # save the model
    if opt.save_model and epoch % 10 == 0:
        torch.save({
            'netD': netD,
            'netG': netG,
            'opt': opt},
            '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
            

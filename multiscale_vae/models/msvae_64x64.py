import torch
import torch.nn as nn

from torch.autograd import Variable

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class conditional_stochastic_encoder(nn.Module):
    def __init__(self, z_dim, y_dim, nc=1, nf=64):
        super(conditional_stochastic_encoder, self).__init__()
        self.y_dim = y_dim
        self.x_encoder = nn.Sequential(
                # input is (nc) x 64 x 64
                dcgan_conv(nc, nf),
                # state size. (nf) x 32 x 32
                dcgan_conv(nf, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_conv(nf * 2, nf * 4),
                # state size. (nf*2) x 8 x 8
        )
        self.y_encoder = nn.Sequential(
                # state size. (y_dim) x 1 x 1
                nn.ConvTranspose2d(y_dim, nf * 4, 4, 1, 0),
                nn.BatchNorm2d(nf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (nf*8) x 4 x 4
                dcgan_upconv(nf * 4, nf * 2),
                # state size. (nf*4) x 8 x 8
            )
        self.combined_encoder = nn.Sequential(
                # state size. (nf*2*2) x 8 x 8
                dcgan_conv(nf * 4 + nf * 2, nf * 4),
                # state size. (nf*4) x 4 x 4
                dcgan_conv(nf * 4, nf * 8),
                # state size. (nf*8) x 2 x 2
        )
       
        self.mu_net = nn.Conv2d(nf * 8, z_dim, 2, 1, 0)
        self.logvar_net = nn.Conv2d(nf * 8, z_dim, 2, 1, 0)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        x, y = input
        x_h = self.x_encoder(x)
        y_h = self.y_encoder(y.view(-1, self.y_dim, 1, 1))
        out = self.combined_encoder(torch.cat([x_h, y_h], 1))
        mu = self.mu_net(out)
        logvar = self.logvar_net(out)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class decoder(nn.Module):
    def __init__(self, z_dim, nc=1, nonlinearity='sigmoid'):
        super(decoder, self).__init__()
        nf = 64
        self.z_dim = z_dim
        outfunc = nn.Sigmoid() if nonlinearity == 'sigmoid' else nn.Tanh()
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(z_dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (nf*8) x 4 x 4
                dcgan_upconv(nf * 8, nf * 4),
                # state size. (nf*4) x 8 x 8
                dcgan_upconv(nf * 4, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_upconv(nf * 2, nf),
                # state size. (nf) x 32 x 32
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                # state size. (nc) x 64 x 64
                outfunc
                )

    def forward(self, input):
        return self.main(input)


class img_conditional_decoder(nn.Module):
    def __init__(self, z_dim, nc=1, nonlinearity='sigmoid'):
        super(img_conditional_decoder, self).__init__()
        nf = 64
        self.z_dim = z_dim
        
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 2 x 2
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, nf * 8, 2, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.Tanh()
                )

        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d((nf * 8) + z_dim, nf * 8, 2, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 2 x 2
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.upc3 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc4 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf) x 16 x 16
        self.upc5 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc6 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                # state size. (nc) x 64 x 64
                )
        if nonlinearity == 'sigmoid':
            self.out = nn.Sigmoid()
        elif nonlinearity == 'tanh':
            self.out = nn.Tanh()
        else:
            self.out = None


    def forward(self, input):
        x, z = input
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)

        z = z.view(-1, self.z_dim, 1, 1)
        d1 = self.upc1(torch.cat([h6, z], 1))
        d2 = self.upc2(torch.cat([d1, h5], 1))
        d3 = self.upc3(torch.cat([d2, h4], 1))
        d4 = self.upc4(torch.cat([d3, h3], 1))
        d5 = self.upc5(torch.cat([d4, h2], 1))
        output = self.upc6(torch.cat([d5, h1], 1))
        if self.out:
            return self.out(output)
        else:
            return output

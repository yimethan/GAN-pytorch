import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from tensorboardX import SummaryWriter

import time
import os

from model.generator import Generator
from model.discriminator import Discriminator
from config import Config

writer = SummaryWriter()

gen = Generator()
dis = Discriminator()

criterion = nn.BCELoss()

if Config.cuda:
    gen.cuda()
    dis.cuda()
    criterion.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = DataLoader(datasets.MNIST('../mnist', train=True, download=True,
                                        transform=transforms.Compose(
                                            [transforms.Resize(Config.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])),
                                        batch_size=Config.batch_size, shuffle=True)

optim_G = Adam(gen.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optim_D = Adam(dis.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

total_steps = len(dataloader) // Config.batch_size * Config.epochs

def log_time(batch_idx, duration, g_loss, d_loss):

    samples_per_sec = Config.batch_size / duration
    training_time_left = (total_steps / step - 1.0) * duration if step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                   " | Gen_loss: {:.5f} | Dis_loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(epoch, batch_idx, samples_per_sec, g_loss, d_loss,
                              sec_to_hm_str(duration), sec_to_hm_str(training_time_left)))

def sec_to_hm_str(t):
    # 10239 -> '02h50m39s'

    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)

step = 0

for epoch in range(Config.epochs):
    for batch_idx, (inputs, _) in enumerate(dataloader):

        start_time = time.time()

        # gt labels
        real = torch.ones(Config.batch_size, 1).to(device)
        fake = torch.zeros(Config.batch_size, 1).to(device)

        inputs = inputs.view(Config.batch_size, -1).to(device)

        # TODO: discriminator

        # loss for real img
        dis_output = dis(inputs)
        dis_real_loss = criterion(dis_output, real)

        # loss for fake img
        x = torch.randn(Config.batch_size, Config.latent_dims).to(device)
        fake_img = gen(x).detach()
        dis_output = dis(fake_img)
        dis_fake_loss = criterion(dis_output, fake)

        # total loss
        dis_total_loss = dis_real_loss + dis_fake_loss

        writer.add_scalar('loss/dis_loss', dis_total_loss.data, epoch)

        dis.zero_grad()
        dis_total_loss.backward()
        optim_D.step()

        # TODO: generator

        x = torch.randn(Config.batch_size, Config.latent_dims).to(device)
        fake_img = gen(x).to(device)
        dis_results = dis(fake_img)

        gen_loss = criterion(dis_results, real)

        writer.add_scalar('loss/gen_loss', gen_loss.data, epoch)

        dis.zero_grad()
        gen.zero_grad()
        gen_loss.backward()
        optim_G.step()

        duration = time.time() - start_time

        gen_path = Config.save_path + '/gen/epoch_{}'.format(epoch)
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        dis_path = Config.save_path + '/dis/epoch_{}'.format(epoch)
        if not os.path.exists(dis_path):
            os.makedirs(dis_path)

        torch.save(gen.state_dict(), gen_path + '/state_dict.pth')
        torch.save(dis.state_dict(), dis_path + '/state_dict.pth')
        torch.save(gen, gen_path + '/model.pth')
        torch.save(dis, gen_path + '/model.pth')

        if batch_idx % 100 == 0:
            print("Epoch: {}/{}, Batch: {}/{}, D loss: {}, G loss: {}".format(epoch, Config.epochs,
                                                                              batch_idx, total_steps,
                                                                              dis_total_loss, gen_loss))

        if batch_idx % Config.log_f == 0:
            log_time(batch_idx, duration, gen_loss.cpu().data, dis_total_loss.cpu().data)

        step += 1

    if epoch % Config.sample_f == 0:

        with torch.no_grad():

            img_path = Config.save_path + '/sample_images/{}'.format(epoch)
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            for img_idx in range(5):
                random_x = torch.randn(5, Config.latent_dims).to(device)
                test_sample = gen(random_x)
                test_sample = test_sample.reshape(5, 280, 280).cpu()

                save_image(test_sample[0], '{}/{}.jpg'.format(img_path, img_idx))

                writer.add_image('sample_imgs/{}/{}'.format(epoch, img_idx), test_sample[0].unsqueeze(0))

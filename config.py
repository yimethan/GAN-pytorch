import torch.cuda

class Config:

    epochs = 10
    batch_size = 16
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    img_size = 280
    
    channels = 1
    latent_dims = 100
    sample_f = 1
    log_f = 125
    save_path = './ckpt'

    img_shape = (channels, img_size, img_size)
    
    cuda = True if torch.cuda.is_available() else False

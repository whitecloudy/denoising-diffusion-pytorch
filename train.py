import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import tensorboard

from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion, Trainer

from absl import app, flags

import Five_G_dataset

#####################################
from torch import nn, einsum
import torch.nn.functional as F
from denoising_diffusion_pytorch.classifier_free_guidance import Downsample, Block

def __main__():
    training_datafiles = ['../ssddata/RENEW/ArgosCSI-96x8-2016-05-01-06-57-58-2.4GHz-continuousmobile', '../ssddata/RENEW/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS']
    validation_datafiles = ["../ssddata/RENEW/ArgosCSI-96x8-2016-05-01-06-38-03-2.4GHz-static"]

    dataset = Five_G_dataset.Five_G_dataset(data_path=training_datafiles, return_complex=False)
    val_dataset = Five_G_dataset.Five_G_dataset(data_path=validation_datafiles, return_complex=False)

    model = Unet(
        dim = 128,
        channels=192,
        condition_channel = (192, 14, 26),
        dim_mults = (1, 2, 2, 2),
        dropout=0.1,
    )

    diffusion = GaussianDiffusion(
        model,
        objective = 'pred_noise',
        image_size = (14, 26),
        beta_schedule='linear',
        timesteps = 1000,    # number of steps
    )

    # classes_emb = nn.Sequential(
    #     Downsample(192, 64),
    #     Downsample(64, 32),
    #     nn.Flatten(),
    # )
    # print(dataset[0][1].unsqueeze(0).shape)
    # print(classes_emb(dataset[0][1].unsqueeze(0)).shape)



    # dataset = CIFAR10(
    #         root='./data', train=True, download=True,
    #         transform=transforms.Compose([
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]))
    
    # print(dataset[0][0].shape)

    import datetime
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_folder: str = "./results/"+current_time
    tensorboard_log_name = './log/snr_test/'+current_time
    
    trainer = Trainer(diffusion, 
                    dataset,
                    validation_dataset=val_dataset,
                    train_batch_size = 128,
                    validation_batch_size= 512,
                    train_lr = 2e-4,
                    train_num_steps = 800000,         # total training steps
                    results_folder=results_folder, # folder to save results
                    gradient_accumulate_every = 1,    # gradient accumulation steps
                    ema_decay = 0.9999,                # exponential moving average decay
                    amp = False,                       # turn on mixed precision
                    save_and_sample_every=25000,       # save and sample every 1000 steps
                    save_best_and_latest_only=True,   # only save the best and the latest model
                    tensorboard_log=tensorboard_log_name,             # log training to tensorboard
                    tensorboard_log_steps=64,         # log training to tensorboard every 100 steps
                    )
    torch.manual_seed(0)
    trainer.train()
    # # after a lot of training

    # sampled_images = diffusion.sample(batch_size = 64)
    # sampled_images.shape # (4, 3, 128, 128)

if __name__=="__main__":
    __main__()

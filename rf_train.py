import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import tensorboard

from denoising_diffusion_pytorch.rf_diffusion import tfdiff_mimo, GaussianDiffusion
from denoising_diffusion_pytorch.classifier_free_guidance import Trainer
from denoising_diffusion_pytorch.params import all_params

from absl import app, flags

import Five_G_dataset

def __main__():
    training_datafiles = ['../ssddata/RENEW/ArgosCSI-96x8-2016-05-01-06-57-58-2.4GHz-continuousmobile', '../ssddata/RENEW/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS']
    # validation_datafiles = ["../ssddata/RENEW/ArgosCSI-96x8-2016-05-01-06-38-03-2.4GHz-static"]
    validation_datafiles = ["../ssddata/RENEW/ArgosCSI-96x2-2016-12-07-03-00-36_rotation_mob_horizontal_omni"]


    dataset = Five_G_dataset.Five_G_dataset(data_path=training_datafiles, transpose=(0, 2, 1))
    val_dataset = Five_G_dataset.Five_G_dataset(data_path=validation_datafiles, transpose=(0, 2, 1))

    params = all_params[2]

    model = tfdiff_mimo(params=params)

    diffusion = GaussianDiffusion(
        model,
        objective = 'pred_x0',
        data_shape = dataset[0][0].shape,
        beta_schedule='linear',
        timesteps = params.max_step,    # number of steps
    )

    import datetime
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    # current_time = "2025-05-24_01-36-22"
    
    tag = "_RF-diffusion_Predict_x0"
    # tag = ""
    results_folder: str = "./results/"+current_time+tag
    tensorboard_log_name = './log/snr_test/'+current_time+tag
    
    trainer = Trainer(diffusion, 
                    dataset,
                    validation_dataset=val_dataset,
                    train_batch_size = 32,
                    validation_batch_size= 64,
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
    # trainer.load("latest")
    torch.manual_seed(0)
    trainer.train()
    # # after a lot of training

    # sampled_images = diffusion.sample(batch_size = 64)
    # sampled_images.shape # (4, 3, 128, 128)

if __name__=="__main__":
    __main__()

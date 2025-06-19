import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import tensorboard

from denoising_diffusion_pytorch.rf_diffusion import tfdiff_mimo, GaussianDiffusion, SignalDiffusion
from denoising_diffusion_pytorch.RF_trainer import Trainer
from denoising_diffusion_pytorch.params import all_params

from absl import app, flags

import Five_G_dataset

def __main__():
    training_datafiles = '../data/RENEW_processed/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS.npy'

    dataset = Five_G_dataset.Five_G_singlefile_dataset(data_path=training_datafiles, time_node_shape=(14, 8), transpose=(0, 2, 1))
    # dataset = Five_G_dataset.Five_G_dataset(data_path=training_datafiles, transpose=(0, 2, 1))
    # val_dataset = Five_G_dataset.Five_G_dataset(data_path=validation_datafiles, transpose=(0, 2, 1))

    training_validation_split_ratio = 0.8

    training_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                                  [int(len(dataset)*training_validation_split_ratio), 
                                                                   len(dataset)-int(len(dataset)*training_validation_split_ratio)], 
                                                                  torch.Generator().manual_seed(0))

    params = all_params[2]

    params.extra_dim = [26, 8]
    params.cond_dim = [26, 8]
    
    model = tfdiff_mimo(params=params)

    # diffusion = GaussianDiffusion(
    #     model,
    #     objective = 'pred_x0',
    #     data_shape = dataset[0][0].shape,
    #     beta_schedule='linear',
    #     timesteps = params.max_step,    # number of steps
    # )
    diffusion = SignalDiffusion(
        model,
        objective = 'pred_x0',
        data_shape = dataset[0][0].shape,
        beta_schedule='linear',
        timesteps = params.max_step,    # number of steps
        freq_blur = params.blur_schedule,  # frequency blur
        use_loss_weights= False,  # do not use loss weights
    )

    import datetime
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    # current_time = "test"
    
    tag = "-Predict_x0-random_split-SignalDiffusion"
    # tag = ""
    results_folder: str = "./results/"+current_time+tag
    tensorboard_log_name = './log/snr_test/'+current_time+tag
    
    trainer = Trainer(diffusion, 
                    training_dataset,
                    validation_dataset=val_dataset,
                    train_batch_size = 64,
                    validation_batch_size= 256,
                    train_lr = 2e-4,
                    train_num_steps = 500000,         # total training steps
                    results_folder=results_folder, # folder to save results
                    gradient_accumulate_every = 1,    # gradient accumulation steps
                    ema_decay = 0.9999,                # exponential moving average decay
                    amp = False,                       # turn on mixed precision
                    save_and_sample_every=5000,       # save and sample every 1000 steps
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

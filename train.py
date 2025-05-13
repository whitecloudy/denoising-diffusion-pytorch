import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import tensorboard

from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion, Trainer

from absl import app, flags

def __main__():
    # FLAGS = flags.FLAGS

    # FLAGS.DEFINE_integer('batch_size', 64, 'Batch size')

    model = Unet(
        dim = 128,
        num_classes = 10,
        dim_mults = (1, 2, 2,2),
        channels=3,
        dropout=0.1,
    )

    diffusion = GaussianDiffusion(
        model,
        objective = 'pred_noise',
        image_size = 32,
        beta_schedule='linear',
        timesteps = 1000,    # number of steps
    )


    dataset = CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    
    trainer = Trainer(diffusion, 
                    dataset,
                    train_batch_size = 128,
                    train_lr = 2e-4,
                    train_num_steps = 800000,         # total training steps
                    gradient_accumulate_every = 1,    # gradient accumulation steps
                    ema_decay = 0.9999,                # exponential moving average decay
                    amp = False,                       # turn on mixed precision
                    calculate_fid = True,              # whether to calculate fid during training
                    num_fid_samples=10000,
                    fid_batch_size=256,
                    save_and_sample_every=50000,       # save and sample every 1000 steps
                    save_best_and_latest_only=True,   # only save the best and the latest model
                    tensorboard_log="./log/pred_noise3",             # log training to tensorboard
                    tensorboard_log_steps=50,         # log training to tensorboard every 100 steps
                    )
    trainer.train()
    # after a lot of training

    sampled_images = diffusion.sample(batch_size = 64)
    sampled_images.shape # (4, 3, 128, 128)

if __name__=="__main__":
    __main__()

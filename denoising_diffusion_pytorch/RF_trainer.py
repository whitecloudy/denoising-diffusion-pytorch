import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter, writer
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from ema_pytorch import EMA
from torchvision import transforms as T, utils

from accelerate import Accelerator, InitProcessGroupKwargs
import accelerate

from einops import rearrange, reduce, repeat, pack, unpack

import tempfile
from pathlib import Path

from datetime import timedelta

from multiprocessing import cpu_count
import copy

from denoising_diffusion_pytorch.version import __version__
from tqdm.auto import tqdm

import re
import html

def read_python_file_cleaned(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. 멀티라인 주석 제거 (''' ''' 또는 """ """)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    code = re.sub(r'"""[\s\S]*?"""', '', code)

    # 2. 한 줄 주석 제거 (// 또는 #)
    code = re.sub(r'#.*', '', code)

    # 3. 여러 줄 공백을 하나의 줄로 압축
    code = re.sub(r'\n\s*\n+', '\n\n', code)

    # 4. 양 끝 공백 제거
    cleaned_code = code.strip()

    # 5. Markdown 코드 블록 포맷으로 감싸기
    markdown_code_block = f"\n```python\n{cleaned_code}\n```"
    return markdown_code_block

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data

def divisible_by(numer, denom):
    return (numer % denom) == 0

# helpers functions
@torch.inference_mode()
def cal_SNR(predict, truth, complex_dim):
    # Recombine the real and imaginary parts to form complex values
    assert predict.shape == truth.shape, "The shapes of predict and truth must match."
    assert predict.shape[complex_dim] % 2 == 0 and truth.shape[complex_dim] % 2 == 0, "The complex dimension must be even."
    real_imag_dim_line = predict.shape[complex_dim]//2
    predict = torch.split(predict, real_imag_dim_line, dim=complex_dim)
    predict_complex = (predict[0] + 1j * predict[1]).squeeze(dim=complex_dim)
    truth = torch.split(truth, real_imag_dim_line, dim=complex_dim)
    truth_complex = (truth[0] + 1j * truth[1]).squeeze(dim=complex_dim)
    PS = torch.sum(torch.abs(truth_complex)**2, dim=(-1, -2, -3))  # power of signal
    PN = torch.sum(torch.abs(predict_complex - truth_complex)**2, dim=(-1, -2, -3))  # power of noise
    ratio = PS / PN
    return 10 * torch.log10(ratio)


class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        validation_dataset = None,
        train_batch_size = 16,
        validation_batch_size = 16,
        validation_active_ratio = None,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_lr_decay = 0.0,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        results_folder = "./results",
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        complex_dim = -1,
        max_grad_norm = 1.,
        # num_fid_samples = 50000,
        save_best_and_latest_only = False,
        tensorboard_log = None,
        tensorboard_log_steps = 100,
    ):
        super().__init__()

        # accelerator

        ipg_handler = InitProcessGroupKwargs(
                    timeout=timedelta(hours=12),
                    )


        self.accelerator = Accelerator(
            kwargs_handlers=[ipg_handler],
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # prepare tensorboard
        self.tensor_board_log_steps = tensorboard_log_steps
        if tensorboard_log is not None and self.accelerator.is_main_process:
            import git
            repo = git.Repo(search_parent_directories=True)
            self.tensor_writer = SummaryWriter(log_dir=tensorboard_log)
            self.tensor_writer.add_text('git_info', 
                                        f'commit: {repo.head.commit.hexsha}\nbranch: {repo.active_branch.name}\ndirty: {repo.is_dirty()}')
            import sys
            self.tensor_writer.add_text('python_info', 
                                        f'python version: {sys.version}\nfile: {read_python_file_cleaned(sys.argv[0])}')
        else:
            self.tensor_writer = None


        # model

        self.model = diffusion_model
        is_ddim_sampling = self.model.is_ddim_sampling

        # sampling and training hyperparameters

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size // self.accelerator.num_processes
        self.gradient_accumulate_every = gradient_accumulate_every
        # assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps

        self.max_grad_norm = max_grad_norm
        # preparing Training dataset and dataloader
        self.ds = dataset

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
        dl = DataLoader(self.ds, 
                        batch_size = self.batch_size,
                        shuffle = True, 
                        pin_memory = True, 
                        num_workers = min(cpu_count()//self.accelerator.num_processes , self.batch_size), # use at most 8 workers
                        persistent_workers=True,)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.complex_dim = complex_dim
        # if self.accelerator.is_main_process:
        #     self.val_ds = validation_dataset

        #     if self.val_ds is not None:
        #         self.val_dl = DataLoader(self.val_ds, batch_size = validation_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())
        #     else:
        #         self.val_dl = None
        # else:
        #     self.val_dl = None
        self.validation_batch_size = validation_batch_size//self.accelerator.num_processes

        # prepare validation dataset and dataloader
        self.val_ds = validation_dataset
        if self.val_ds is not None:
            self.val_dl = DataLoader(self.val_ds, batch_size = self.validation_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count()//self.accelerator.num_processes)

            self.val_dl = self.accelerator.prepare(self.val_dl)

            if validation_active_ratio is not None:
                self.active_val_len = int(len(self.val_dl) * validation_active_ratio)
            else:
                self.active_val_len = len(self.val_dl)


            self.dummy_ema_model = copy.deepcopy(self.model)
            self.dummy_ema_model.eval()
            self.dummy_ema_model.requires_grad_(False)
            self.dummy_ema_model.to(self.device)
            self.dummy_ema_model.tqdm_disable = not self.accelerator.is_main_process
        else:
            self.val_dl = None

        # optimizer

        self.opt = Adam(self.model.parameters(), lr = train_lr, betas = adam_betas, weight_decay=train_lr_decay)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        if results_folder is None:
            self.result_temp_folder = tempfile.TemporaryDirectory()
            self.results_folder = Path(self.result_temp_folder.name)
        else:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if save_best_and_latest_only:
            self.best_SNR = -1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        total_loss = 0.
        cum_loss = None

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                for _ in range(self.gradient_accumulate_every):
                    data, cond = next(self.dl)
                    data = data.to(device, non_blocking = True)
                    cond = cond.to(device, non_blocking = True)

                    with self.accelerator.autocast():
                        loss = self.model(data, classes=cond)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                if cum_loss is None:
                    cum_loss = loss.item()
                else:
                    cum_loss  = cum_loss*0.9 + loss.item()*0.1
                pbar.set_description(f'loss: {cum_loss:.4f}')

                self.step += 1

                self.grad_norm = accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                if (self.tensor_writer is not None) and (self.step % self.tensor_board_log_steps == 0):
                    self.tensor_writer.add_scalar('Train/loss', total_loss/self.tensor_board_log_steps, self.step)
                    self.tensor_writer.add_scalar('Train/grad_norm', self.grad_norm, self.step)
                    total_loss = 0.

                pbar.update(1)
                if accelerator.is_main_process:
                    self.ema.update()

                # Save and validate
                if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    with torch.inference_mode():
                        # Validation
                        if self.val_dl is not None:
                            SNR_sum = torch.tensor(0.).to(device)
                            loss_sum = torch.tensor(0.).to(device)
                            test_data_len = torch.tensor(0.)

                            # Broadcast ema model state dict
                            if self.accelerator.is_main_process:
                                dummy_ema_state = self.ema.ema_model.state_dict()
                            else:
                                dummy_ema_state = self.dummy_ema_model.state_dict()

                            accelerate.utils.broadcast(dummy_ema_state)
                            self.dummy_ema_model.load_state_dict(dummy_ema_state)

                            # Validation loop
                            for v_idx, (data, cond) in enumerate(tqdm(self.val_dl, total= self.active_val_len, desc = 'validation loop', disable = not accelerator.is_main_process)):
                                data = data.to(device, non_blocking = True)
                                cond = cond.to(device, non_blocking = True)

                                predict = self.dummy_ema_model.sample_with_class(
                                    classes = cond,
                                )

                                loss = F.mse_loss(predict, data, reduction = 'none')
                                loss = reduce(loss, 'b ... -> b', 'mean')
                                loss_sum += torch.sum(loss)

                                SNR = cal_SNR(predict, data, complex_dim = self.complex_dim)
                                SNR_sum += torch.sum(SNR)

                                test_data_len += data.shape[0]

                                if v_idx >= self.active_val_len:
                                    break
                            
                            gathered_SNR = accelerator.gather_for_metrics(SNR_sum)
                            gathered_loss = accelerator.gather_for_metrics(loss_sum)
                            gathered_len = accelerator.gather_for_metrics(test_data_len.to(device))
                            if accelerator.is_main_process:
                                gathered_len = torch.sum(gathered_len).item()
                                SNR = torch.sum(gathered_SNR).item() / gathered_len
                                loss = torch.sum(gathered_loss).item() / gathered_len
                                accelerator.print(f'SNR: {SNR:.2f}, Loss: {loss:.4f}')
                                if self.tensor_writer is not None:
                                    self.tensor_writer.add_scalar('Validation/SNR', SNR, self.step)
                                    self.tensor_writer.add_scalar('Validation/Loss', loss, self.step)
                        
                        # save model
                        milestone = self.step // self.save_and_sample_every
                        if self.accelerator.is_main_process:
                            if self.save_best_and_latest_only:
                                if self.best_SNR < SNR:
                                    self.best_SNR = SNR
                                    self.save("best")
                                self.save("latest")
                            else:
                                self.save(milestone)

                        if self.tensor_writer is not None:
                            self.tensor_writer.flush()

                        accelerator.wait_for_everyone()

        accelerator.print('training complete')

    def __del__(self):
        if self.tensor_writer is not None:
            self.tensor_writer.close()
        self.accelerator.end_training()

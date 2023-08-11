from math import ceil
import time
from typing import Mapping, Tuple
import os

import torch

from grokking.data import get_data
from grokking.model import Transformer
from grokking.logger import Logger, DEFAULT_WANDB_METRICS
from grokking import utils
from grokking.utils import ExponentialMovingAverage, SmoothedDyDx

s1=0
exp_name = 'log_p2-1_e500'
def evaluate(model, val_loader, device, criterion)->Tuple[float, float]:
    correct = 0
    loss_sum = 0.
    loss, acc = 0., 0.

    global s1
    s1+=1

    pass1, pass2 = 2, 1 # 1, 2 is low score, 2, 2 # high score

    if s1<pass1:
        return -1, -1

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Loop over each batch from the validation set
        for batch in val_loader:
            if s1>=pass2:
                inputs, labels = tuple(t.to(device) for t in batch)
                #inputs, labels = torch.unsqueeze(inputs,0), torch.unsqueeze(labels,0)

                # a=inputs[:,1]
                # b=inputs[:,3]
                # assert(sum(((labels-11)*(b-11) % 223)==(a-11))==(len(labels)))

                output = model(inputs)[-1,:,:]
                correct += (torch.argmax(output, dim=1) == labels).sum().item()
                loss_sum += criterion(output, labels).item() * len(labels)
            #assert count == len(val_loader.dataset)

    loss = loss_sum / len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)

    model.train()
    return loss, acc


def train(config:Mapping):
    if not config['device']:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_name = config['device']

    out_dir = utils.full_path(config['out_dir'], create=True)

    utils.setup_torch()
    utils.setup_seed(config['seed'])

    logger = Logger(log_filepath=os.path.join(out_dir, exp_name+'.txt'),
                    enable_wandb=config['use_wandb'], master_process=True,
                    wandb_project=config['wandb_project'], wandb_run_name=config['wandb_run'],
                    config=config,
                    wandb_metrics=DEFAULT_WANDB_METRICS + [
                        {"name": "train/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "val/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "lr", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "ETA_hr", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "w_norm", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "train/d_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "val/d_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "train/ewa_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "val/ewa_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "w_norm_ewa", "step_metric":"train/step", "summary":"min", "goal":"min"},
                    ])

    logger.summary(config)

    device = torch.device(device_name)
    num_steps = config['num_steps']
    eval_every = config['eval_every']

    # get dataset
    start_time = time.time()
    train_loader, val_loader, tokenizer = get_data(
        config['operation'],
        config['prime'],
        config['training_fraction'],
        config['batch_size'],
        config['eval_batch_size'],
    )
    data_gen_time = time.time() - start_time

    # create model
    model = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=len(tokenizer),
        seq_len=5, # currently each input eq has [eos a op b =] which is 5 tokens
        ).to(device)

    logger.summary({"device_name": device_name,
                    'train_data_len': len(train_loader.dataset),
                    'val_data_len': len(val_loader.dataset),
                    'train_loader_len': len(train_loader),
                    'val_loader_len': len(val_loader),
                    'epochs': ceil(num_steps / len(train_loader)),
                    'model_params': model.get_num_params(True),
                    'model_params_all': model.get_num_params(False),
                    'vocab_size': len(tokenizer),
                    'data_gen_time_s': data_gen_time,})

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        weight_decay=config['weight_decay']
        )

    # scheduler provides warmup and then constant lr
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 1.e-8, total_iters=10
    )

    step, start_time = 0, time.time()
    ewa_train_loss, ewa_val_loss = ExponentialMovingAverage(weight=0.1), ExponentialMovingAverage(weight=0.1)
    w_norm_ewa = ExponentialMovingAverage(weight=0.3)
    d_train_loss = SmoothedDyDx(y_ema_weight=1.0, x_ema_weight=0.5, dx_ema_weight=0.9,
                                 dy_ema_weight=1.0, dydx_ema_weight=0.1)
    d_val_loss = SmoothedDyDx(y_ema_weight=1.0, x_ema_weight=0.5, dx_ema_weight=0.9,
                                 dy_ema_weight=1.0, dydx_ema_weight=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    while step < num_steps:
        # Loop over each batch from the training set
        for batch in train_loader:
            inputs, labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # model output is tensor [5,batch_size,prime+2]
            # [EOS a op b =] is input to model which is 5 tokens
            # output is [a op b = c] which is 5 tokens
            # we only take the last token of the output for loss
            output = model(inputs)[-1,:,:]
            loss = criterion(output, labels)
            acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            ewa_train_loss.add(loss.item())
            d_train_loss.add(loss.item(), step)

            if step % 20 == 0 or step+1 >= num_steps:
                metrics = {
                    "train/step": step,
                    "train/acc": acc.item(),
                    "train/loss": loss.item(),
                    "train/ewa_loss": ewa_train_loss.value,
                    "train/d_loss": d_train_loss.value,
                }
                #logger.info(metrics)

            if step % eval_every == 0 or step+1 >= num_steps:
                val_loss, val_acc = evaluate(model, val_loader, device, criterion)

                # ewa_val_loss.add(val_loss)
                # d_val_loss.add(val_loss, step)

                # w_norm = model.weight_norm()
                # w_norm_ewa.add(w_norm)

                val_metrics = {
                    "train/step": step,
                    "val/acc": val_acc,
                    # "val/loss": val_loss,
                    # "w_norm": w_norm,
                    # "w_norm_ewa": w_norm_ewa.value,
                    # "ETA_hr": (time.time() - start_time) / (step+1) * (num_steps - step) / 3600,
                    # "lr": optimizer.param_groups[0]['lr'],
                    # "val/ewa_loss": ewa_val_loss.value,
                    # "val/d_loss": d_val_loss.value,
                }
                logger.info(val_metrics)

            step += 1
            if step >= num_steps:
                break

    logger.summary({"train_time_hr": (time.time() - start_time)/3600,
                    "step_time_s": (time.time() - start_time)/step,})
    logger.finish()
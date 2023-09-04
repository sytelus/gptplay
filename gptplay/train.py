from contextlib import nullcontext
from typing import Mapping, Tuple
import dataclasses
import math
import numpy as np
import random

import torch
from torch.nn.parallel import DistributedDataParallel

from gptplay import utils


@torch.no_grad()
def estimate_loss(model, get_loss, data_loader, eval_iters, is_cuda, device)->Tuple[float, float]:
    model.eval()
    loss_sum, correct_sum, data_count = 0., 0, 0
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device) if is_cuda else x.to(device), \
               y.to(device) if is_cuda else y.to(device)
        logits = model(x)
        loss, correct = get_loss(logits, y)
        loss_sum += loss.item() * len(y)
        correct_sum += correct.item()
        data_count += len(y)
    model.train()
    return loss_sum / data_count, correct_sum / data_count

def log_metrics(logger, step, model, get_loss, eval_iters, lr,
                is_cuda, device, train_loader, val_loader, test_loader, seed):

    train_loss, train_acc = estimate_loss(model, get_loss, train_loader, eval_iters,
                                    is_cuda, device)

    val_loss, val_acc = estimate_loss(model, get_loss, val_loader, eval_iters,
                                    is_cuda, device)

    w_norm = model.weight_norm()

    metrics = {
        "seed": seed,
        "train/step": step,
        "train/loss": train_loss,
        "train/ppl": math.exp(train_loss),
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/ppl": math.exp(val_loss),
        "val/acc": val_acc,
        "w_norm": w_norm,
        "lr": lr,
    }

    if test_loader:
        test_loss, test_acc = estimate_loss(model, get_loss, test_loader, eval_iters,
                                    is_cuda, device)
        metrics["test/loss"] = test_loss,
        metrics["test/ppl"] = math.exp(test_loss),
        metrics["test/acc"] = test_acc,

    logger.info(metrics)

    return val_loss

def clean(config:Mapping)->Mapping:
    """Remove module key from config so we can pass it as arguments to functions."""
    c = config.copy()
    c.pop('module')
    return c

def train(config:Mapping, logger):
    project_name = config['logging']['project_name']
    run_name = config['logging']['run_name']
    device_type = config['general']['device_type']
    dtype = config['general']['dtype']
    enable_distributed = config['general']['enable_distributed']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    train_batch_size = config['data']['train_batch_size']
    seed = config['general']['seed']
    torch_compile = config['general']['torch_compile']
    num_steps = config['training']['num_steps']
    grad_clip = config['training']['grad_clip']
    train_log_every = config['training']['log_every']
    eval_every = config['eval']['eval_every']
    eval_iters = config['eval']['eval_iters']
    save_checkpoint = config['eval']['save_checkpoint']
    checkpoint_every = config['eval']['checkpoint_every']
    checkoint_after = config['eval']['checkoint_after']
    out_dir = config['general']['out_dir']
    data_config = config['data']
    model_config = config['model']
    context_length = config['model']['context_length']
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    tokenizer_config = config['tokenizer']

    get_data = utils.import_fn(config['data']['module'])
    get_tokenizer = utils.import_fn(config['tokenizer']['module'])
    get_optim = utils.import_fn(config['optimizer']['module'])
    get_scheduler = utils.import_fn(config['scheduler']['module'])
    get_model = utils.import_fn(config['model']['module'])
    get_loss = utils.import_fn(config['loss']['module'])


    def setup_torch():
        # show Tensor shape first for tensor's rpresentation
        normal_repr = torch.Tensor.__repr__
        torch.Tensor.__repr__ = lambda self: f"{tuple(self.shape)}:{normal_repr(self)}" # type: ignore

        torch.backends.cudnn.enabled = True
        torch.set_printoptions(precision=10)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    setup_torch()
    setup_seed(seed)

    # logger.summary(dataclasses.asdict(torch_info))
    # logger.summary({"global_batch_size": gradient_accumulation_steps * train_batch_size * torch_info.world_size,
    #                 "local_batch_size": gradient_accumulation_steps * torch_info.world_size,
    #                 "tokens_per_iter": gradient_accumulation_steps * train_batch_size * torch_info.world_size * context_length
    #                 })

    device = torch.device('cuda')

    # get dataset
    train_loader, val_loader, test_loader = get_data(local_rank=0,
                                                     **clean(data_config))
    tokenizer = get_tokenizer(**clean(tokenizer_config))

    # logger.summary({'vocab_size': len(tokenizer),
    #                 'train_len': len(train_loader.dataset),
    #                 'val_len': len(val_loader.dataset),
    #                 'test_len': len(test_loader.dataset) if test_loader is not None else 0,
    #                 'train_batches': len(train_loader),
    #                 'val_batches': len(val_loader),
    #                 'test_batches': len(test_loader) if test_loader is not None else 0
    #                 })

    # create model
    model = get_model(vocab_size=len(tokenizer),
                      **clean(model_config)).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['learning_rate'],
        betas=(0.9, 0.98),
        weight_decay=config['optimizer']['weight_decay']
        )


    if torch_compile:
        logger.info("Compiling model...")
        try:
            model = torch.compile(model) # requires PyTorch 2.0
        except Exception as e:
            logger.error(f"Failed to compile model: {str(e)}")
        logger.info("Compiling done.")


    # scheduler provides warmup and then constant lr
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 1.e-8, total_iters=10
    )

    step, eval_count = 0, 0
    epoch, epoch_step = 0, 0 # epoch steps is useful to know how many epochs we did
    best_val_loss, evel_count = float('inf'), 0

    out_dir = utils.full_path(out_dir, create=True)
        #logger.info({'out_dir': out_dir})

    # run steps
    while step < num_steps:
        epoch_step = 0 # step within the epoch

        # Loop over the training set
        for t in train_loader:
            model.train()
            optimizer.zero_grad()

            x, y = tuple(t)
            x, y = x.to(device), \
                   y.to(device)

            loss_sum, correct_sum, data_count = 0., 0, 0
            logits = model(x)
            loss, correct = get_loss(logits, y)

            # backward pass, with gradient scaling if training in fp16
            loss.backward()

            # clip the gradient
            # if grad_clip != 0.0:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # step the optimizer and scaler if training in fp16
            optimizer.step()
            # scaler.update()
            scheduler.step()
            # flush the gradients as soon as we can, no need for this memory anymore

            loss_sum += loss.item() * len(y)
            correct_sum += correct.item()
            data_count += len(y)

            # log train loss for this step
            # if torch_info.is_master and (step+1) % train_log_every == 0 or (step+1) >= num_steps:
            #     metrics = {
            #         "train/step": step,
            #         "train/step_acc": correct_sum / data_count,
            #         "train/step_loss": loss_sum / data_count,
            #         "train/step_ppl": math.exp(loss_sum / data_count),
            #     }
            #     logger.info(metrics)

            # log eval metrics upto this step
            if (step+1) % eval_every == 0 or step+1 >= num_steps:
                val_loss = log_metrics(logger, step, model, get_loss, eval_iters,
                    optimizer.param_groups[0]['lr'],
                    True, device, train_loader, val_loader,
                    test_loader if step+1 >= num_steps else None, seed)

            step += 1
            epoch_step += 1
            if step >= num_steps:
                break

        epoch += 1


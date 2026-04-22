from functools import partial

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil

from accelerate import Accelerator
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from contextlib import nullcontext

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.dataset.pusht_image_dataset import SourceTargetDataset


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # =========================================================
        # 提前锁定多卡通信环境，防止稍后 load_weight 污染 CUDA 上下文
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs, ddp_kwargs])
        # =========================================================

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.pretrained_ckpt = cfg.pretrained_ckpt if hasattr(cfg, 'pretrained_ckpt') else None
        if self.pretrained_ckpt and os.path.exists(self.pretrained_ckpt):
            self.model.load_weight_from_ckpt(self.pretrained_ckpt, use_ema=True)  # NOTE: ori:False

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        else:
            print(f"[TrainDiffusionTransformerHybridWorkspace] training from scratch")

        # freeze some parameters if needed
        if getattr(cfg.policy, 'en_freeze_obs_encoder', False):
            print("[TrainDiffusionTransformerHybridWorkspace] Freezing obs_encoder...")
            if hasattr(self.model, 'freeze_obs_encoder'):
                self.model.freeze_obs_encoder()
        self.model.print_training_status()

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # Domain adaptation
        self.is_da = hasattr(cfg.policy, 'use_da') and cfg.policy.use_da
        print(f"[TrainDiffusionTransformerHybridWorkspace] is_da: {self.is_da}")

    def run(self):  # Main Entry
        cfg = copy.deepcopy(self.cfg)

        # =========================================================
        # 直接获取在 __init__ 中安全初始化好的 accelerator
        accelerator = self.accelerator
        device = accelerator.device
        # =========================================================

        # resume training
        if cfg.training.resume:
            # 注意：多卡模式下，只需主进程打印
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file() and accelerator.is_main_process:
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
            # 模型加载需在所有进程同步执行
            if lastest_ckpt_path.is_file():
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # (DA) target dataset
        if self.is_da:
            dataset_tgt: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset_target)
            assert isinstance(dataset, BaseImageDataset)
            dataset_src_tgt = SourceTargetDataset(dataset, dataset_tgt)
            train_dataloader = DataLoader(dataset_src_tgt, drop_last=True, **cfg.dataloader)

        # ## Debug dataset
        # dataset.__getitem__(dataset.__len__() - 1)
        # dataset_tgt.__getitem__(dataset_tgt.__len__() - 1)
        # dataset_src_tgt.__getitem__(dataset_src_tgt.__len__() - 1)
        # print(len(dataset), len(dataset_tgt), len(dataset_src_tgt))
        # exit()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, drop_last=True, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        if self.is_da:
            (g_vis1_opt, d_vis1_opt,
             g_vis2_opt, d_vis2_opt,
             g_act_opt, d_act_opt) = self.optimizer
            get_sch = partial(get_scheduler,
                              name=cfg.training.lr_scheduler,
                              num_warmup_steps=cfg.training.lr_warmup_steps,
                              num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) \
                                                 // cfg.training.gradient_accumulate_every,
                              # pytorch assumes stepping LRScheduler every epoch
                              # however huggingface diffusers steps it every batch
                              last_epoch=self.global_step-1)
            g_vis1_sch = get_sch(optimizer=g_vis1_opt)
            d_vis1_sch = get_sch(optimizer=d_vis1_opt)
            g_vis2_sch = get_sch(optimizer=g_vis2_opt)
            d_vis2_sch = get_sch(optimizer=d_vis2_opt)
            g_act_sch = get_sch(optimizer=g_act_opt)
            d_act_sch = get_sch(optimizer=d_act_opt)
            lr_schedulers = (g_vis1_sch, d_vis1_sch,
                             g_vis2_sch, d_vis2_sch,
                             g_act_sch, d_act_sch)
        else:
            self.lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
                                   // cfg.training.gradient_accumulate_every,
                last_epoch=self.global_step - 1
            )

        # ### [修改点 3] 让 Accelerator 接管模型、优化器和 DataLoader ###
        # 被接管后，accelerator 会自动处理 .to(device) 和 DDP 的包装
        if self.is_da:
            prepared = accelerator.prepare(
                self.model, train_dataloader, val_dataloader,
                *self.optimizer, *lr_schedulers
            )
            self.model = prepared[0]
            train_dataloader = prepared[1]
            val_dataloader = prepared[2]
            self.optimizer = tuple(prepared[3:9])
            lr_schedulers = tuple(prepared[9:])
        else:
            self.model, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler = accelerator.prepare(
                self.model, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler
            )

        # configure ema (EMA 模型通常不需要放入 DDP 包装中，手动放到 device 即可)
        ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema_model.to(device)
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env (只有主进程需要跑仿真 Rollout)
        env_runner = None
        wandb_run = None
        topk_manager = None

        if accelerator.is_main_process:
            if cfg.training.rollout_every != -1:
                env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

            # To avoid time inconsistency between ckpt dir and wandb run name
            run_dir = pathlib.Path(self.output_dir)  # data/outputs/YYYY.MM.DD/HH.MM.SS_xxx
            logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            logging_cfg["name"] = f"{run_dir.parent.name}-{run_dir.name}"  #YYYY.MM.DD - HH.MM.SS_xxx

            # Save dataset statistics if supported
            if hasattr(dataset, "compute_statistics_and_save_json"):
                stats = dataset.compute_statistics_and_save_json(os.path.join(run_dir, "statistics.json"))
                from robokit.debug_utils.printer import beautiful_print
                print(f"[DEBUG] statistics saved. Content is:")
                beautiful_print(stats)

            print(f"[DEBUG] out_dir is: {run_dir}")

            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **logging_cfg
            )
            wandb.config.update({"output_dir": self.output_dir})

            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

            log_path = os.path.join(self.output_dir, 'logs.json.txt')
            log_ctx = JsonLogger(log_path)
        else:
            log_ctx = nullcontext(None)
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        # ### [修改点 4] 仅主进程开启 tqdm ###
        disable_tqdm = not accelerator.is_main_process
        with log_ctx as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_losses = list()

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=disable_tqdm) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        losses_log = {}

                        if not self.is_da:
                            # Vanilla training
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            with accelerator.accumulate(self.model):
                                losses: dict = self.model(batch)
                                raw_loss = losses['total_loss']
                                loss = raw_loss
                                accelerator.backward(loss)

                                if accelerator.sync_gradients:
                                    self.optimizer.step()
                                    self.lr_scheduler.step()
                                    self.optimizer.zero_grad()

                                for k, v in losses.items():
                                    if k == 'total_loss': continue
                                    losses_log[k] = v.item() if isinstance(v, torch.Tensor) else v
                        else:
                            # Domain adaptation
                            batch_src = batch['src']
                            if train_sampling_batch is None:
                                train_sampling_batch = batch_src

                            # ### [致命隐患修正注意] ###
                            # 你的 self.model 是被 accelerator 包装过的。
                            # 如果 compute_loss 内部有 .backward()，你必须把 accelerator 作为参数传进去，
                            # 并在 compute_loss 内部用 accelerator.backward(loss) 替换原有的 loss.backward()，否则多卡必卡死！
                            # losses: dict = self.model.compute_loss(
                            #     batch, self.optimizer, lr_schedulers, batch_idx=batch_idx,
                            #     accelerator=accelerator  # <- 强烈建议传入 accelerator 以便在内部调用
                            # )
                            losses: dict = self.model(
                                batch, optimizers=self.optimizer, lr_schedulers=lr_schedulers,
                                batch_idx=batch_idx, accelerator=accelerator
                            )
                            raw_loss = losses['da_d1_loss']
                            for k, v in losses.items():
                                losses_log[k] = v.item() if isinstance(v, torch.Tensor) else v
                            lr_scheduler = lr_schedulers[0]

                        # ### [修改点 6] 获取真实的 model 进行 EMA 更新 ###
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # ### [修改点 7] 仅主进程进行 Logging ###
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)

                        if accelerator.is_main_process:
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': self.lr_scheduler.get_last_lr()[0],
                                **losses_log
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                            if not is_last_batch:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                        self.global_step += 1

                # ========= 评测阶段 (仅让主进程执行 Rollout 和保存，避免冲突) =========
                # 同步所有 GPU，确保大家都跑完了这个 Epoch
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    step_log['train_loss'] = np.mean(train_losses)

                    # Unwrap model to pass to runner
                    unwrapped_policy = accelerator.unwrap_model(self.model)
                    policy = unwrapped_policy
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    # run rollout
                    if cfg.training.rollout_every != -1 and ((self.epoch) % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)

                # run validation (所有卡都可以参与计算验证集，但最简单的做法是只让主卡算，或者算完 gather。此处保持原逻辑但在主卡上收集)
                if (self.epoch % cfg.training.val_every) == 0:
                    policy_val = self.ema_model if cfg.training.use_ema else accelerator.unwrap_model(self.model)
                    policy_val.eval()

                    val_loss_sum = torch.tensor(0.0, device=device)
                    val_batches = torch.tensor(0, device=device)

                    with torch.no_grad():
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                                disable=disable_tqdm,
                        ) as tepoch:
                            for batch in tepoch:
                                loss = policy_val.compute_loss(batch)
                                loss = loss.detach() if isinstance(loss, torch.Tensor) else loss["total_loss"].detach()
                                val_loss_sum += loss
                                val_batches += 1

                    stats = torch.stack([val_loss_sum, val_batches.float()])
                    stats = accelerator.reduce(stats, reduction="sum")

                    if accelerator.is_main_process and stats[1] > 0:
                        step_log["val_loss"] = (stats[0] / stats[1]).item()

                # run diffusion sampling
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        obs_dict = train_sampling_batch['obs']
                        gt_action = train_sampling_batch['action']
                        result = policy.predict_action(obs_dict)
                        mse = torch.nn.functional.mse_loss(result['action_pred'], gt_action)
                        step_log['train_action_mse_error'] = mse.item()

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()

                        # DDP 下不建议保存 snapshot；只在单进程时保留
                        if cfg.checkpoint.save_last_snapshot and accelerator.num_processes == 1:
                            self.save_snapshot()

                        metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                        # NOTE: why no val_loss? {'train_loss': 0.44689754078118626, 'global_step': 282, 'epoch': 0, 'lr': 5.66e-05, 'train_action_mse_error': 0.2813175352475294}
                        monitor_key = cfg.checkpoint.topk.monitor_key
                        if monitor_key not in metric_dict:
                            print(f"[WARN] skip topk ckpt: monitor_key `{monitor_key}` not in step_log.")
                        else:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                            if topk_ckpt_path is not None:
                                self.save_checkpoint(path=topk_ckpt_path)

                if accelerator.is_main_process:
                    policy.train()
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                self.epoch += 1
                accelerator.wait_for_everyone()  # 确保下一轮开始前所有进程同步


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

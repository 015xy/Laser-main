"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path
import pandas as pd
from typing import List
from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
from common.dist_utils import (
    download_cached_file,
    is_main_process,
    main_process,
)
from common.registry import registry
from common.utils import is_url
from datasets.datasets.dataloader_utils import MultiIterLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ChainDataset

from torch.cuda.amp import autocast

IMAGE_TOKEN_ID = 101

class IDataset(Dataset):
    def __init__(self, path, use_cf=False, short_prompt=False):
        self.path = path
        self.data = pd.read_csv(self.path)
        self.use_cf = use_cf
        self.short_prompt = short_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data.iloc[idx]
        input = d['input']

        if self.short_prompt:
            # input = input.replace("You are an intelligent recommendation assistant. Please summarize my characteristics with a single token based on my browsing history. In chronological order, I have browsed the following items:",
            #                           "Summarize the item characteristics with a single token:")
            input = input.replace("You are an intelligent recommendation assistant. Please summarize the item's characteristics with a single token:",
                            "")

        if not self.use_cf:
            return input
        else:
            return input, \
                torch.tensor(eval(d['input_embedding']), dtype=torch.float32), \
                torch.tensor(eval(d['input_mask']), dtype=int)
            
class ValidDataset(Dataset):
    def __init__(self, path, use_cf=False, short_prompt=False):
        self.path = path
        self.data = pd.read_csv(self.path)
        self.use_cf = use_cf
        self.short_prompt = short_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data.iloc[idx]
        input = d["input"]

        if self.short_prompt:
            input = input.replace("You are an intelligent recommendation assistant. Please summarize my characteristics with a single token based on my browsing history. In chronological order, I have browsed the following items:",
                                "")

        if not self.use_cf:
            return input, d['label']
        else:
            return input, d['label'],\
            torch.tensor(eval(d['input_embedding']), dtype=torch.float32), \
            torch.tensor(eval(d['input_mask']), dtype=int)
    
def collate_fn(batch):
    
    return batch

def collate_fn_use_cf(batch):
    descriptions = [item[0] for item in batch]
    embs = [item[1] for item in batch]  
    masks = [item[2] for item in batch] 
    
    return descriptions, torch.stack(embs, dim=0), torch.stack(masks, dim=0) 

def collate_fn_for_eval(batch):
        
    descriptions = [item[0] for item in batch]
    labels = [item[1] for item in batch]  
    return descriptions, labels

def collate_fn_for_eval_use_cf(batch):
        
    descriptions = [item[0] for item in batch]
    labels = [item[1] for item in batch]  
    
    input_embs = [item[2] for item in batch]  
    input_masks = [item[3] for item in batch]  

    return descriptions, labels, torch.stack(input_embs, dim=0), torch.stack(input_masks, dim=0)

@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.iters_per_epoch = self.datasets[list(self.datasets.keys())[0]]['train'].row_count // self.config.run_cfg.batch_size_train
        self.output_dir = cfg.run_cfg.output_dir

        self.best_ndcg = 0.0
        self.best_epoch = 0

    @property
    def device(self):
        if self._device is None:
            # self._device = [torch.device(device) for device in self.config.run_cfg.device]
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu], find_unused_parameters=True
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %s" % format(num_parameters))
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)
            max_epoch = self.max_epoch
            min_lr = self.min_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            self.datasets = self.datasets[list(self.datasets.keys())[0]]
            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.args.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        # if len(valid_splits) == 0:
        #     logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]
        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()

        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        if self.config.model_cfg.finetune_stage == "2-3": 
            ckp = torch.load(self.config.model_cfg.pretrained)
            if "I" in ckp.keys():
                self.model.I = ckp['I'].to(self.config.run_cfg.I_device)
            else:
                self.obtain_I(self.config.model_cfg.use_cf, self.config.model_cfg.short_prompt)
                
        for cur_epoch in range(self.start_epoch, self.max_epoch):
            logging.info(f"epoch {cur_epoch}")
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
            # evaluation phase
            if self.config.model_cfg.arch == 'translator_arxiv': 
                self._save_checkpoint(cur_epoch, stage_1=True)
            else:
                
                logging.info("Start testing")
                metrics = self.eval_epoch(self.config.run_cfg.test_data_dir, self.config.model_cfg.use_cf, self.config.model_cfg.short_prompt)
      
            if self.evaluate_only:
                break

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def prepare_lm_input(self, texts: List[str]):

        PREFIX_TOKEN_ID = 101
        SUFFIX_TOKEN_ID = 102

        # 获得input_ids
        input_ids = self.model.chatglm2_tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt').input_ids.to(self.model.device)
        
        # 获得input_embs
        input_embs = self.model.chatglm2_model.transformer.embedding.word_embeddings(input_ids)

        batch_size = len(texts)
        suffix = self.model.suffix.expand(batch_size, -1, -1)

        input_embs = torch.cat([input_embs, suffix], dim=1)

        if self.model.config.use_prefix:
            prefix = self.model.prefix.expand(batch_size, -1, -1)
            input_embs = torch.cat([prefix, input_embs], dim=1)

        # 更新input_ids
        suffix_ids = torch.tensor([SUFFIX_TOKEN_ID] * batch_size).view(batch_size, 1).to(self.model.device)
        input_ids = torch.cat([input_ids, suffix_ids], dim=1)

        if self.model.config.use_prefix:
            prefix_ids = torch.tensor([PREFIX_TOKEN_ID] * batch_size * self.model.config.num_query_token)\
                            .view(batch_size, self.model.config.num_query_token).to(self.model.device)
            input_ids = torch.cat([prefix_ids, input_ids], dim=1)

        # 返回input_ids和input_embs
        return input_ids, input_embs.transpose(0, 1).contiguous()

    @torch.no_grad()
    def obtain_I(self, use_cf=False):
        self.model.eval()

        ### (1) 创造dataset 和 dataloader
        item_emb_path = self.config.run_cfg.item_emb_path
        I_dataset = IDataset(item_emb_path, use_cf, self.config.model_cfg.short_prompt)
        if not use_cf:
            I_dataloader = DataLoader(I_dataset, batch_size=self.config.run_cfg.batch_size_eval, \
                                        shuffle=False, num_workers=self.config.run_cfg.num_workers, collate_fn=collate_fn)
        else:
            I_dataloader = DataLoader(I_dataset, batch_size=self.config.run_cfg.batch_size_eval, \
                            shuffle=False, num_workers=self.config.run_cfg.num_workers, collate_fn=collate_fn_use_cf)
            
        ### (2) 输入模型，获得I
        if not use_cf:
            self.model.I = []
            for data in tqdm(I_dataloader, desc="obtain I"):
                input_ids, input_embeds = self.prepare_lm_input(data)
                with autocast():
                    output_features = self.model.chatglm2_model(
                        input_ids=input_ids,
                        inputs_embeds=input_embeds,
                        return_dict=True
                    )
                self.model.I.append(output_features.to(self.config.run_cfg.I_device))
            self.model.I = torch.cat(self.model.I, dim=0)
        else:
            self.model.I = []
            for data in tqdm(I_dataloader, desc="obtain I"):
                text = data[0]  
                embs = data[1]  
                masks = data[2]  
                with autocast():
                    item_features, _ = self.model.obtain_features(
                        item_embs=embs, 
                        item_atts=masks, 
                        text=text
                    )
                self.model.I.append(item_features.to(self.config.run_cfg.I_device))
            self.model.I = torch.cat(self.model.I, dim=0)

    def train_epoch(self, epoch):
        # 2-1/3阶段
        if self.config.model_cfg.finetune_stage in ["2-1", "2-3"]:
            self.model.train()
            return self.task.train_epoch(
                epoch=epoch,
                iters_per_epoch=self.iters_per_epoch,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                lr_scheduler=self.lr_scheduler,
                cuda_enabled=self.cuda_enabled,
                log_freq=self.log_freq,
                accum_grad_iters=self.accum_grad_iters,
            )

        # 2-2阶段
        elif self.config.model_cfg.finetune_stage == "2-2":
            self.obtain_I(self.config.model_cfg.use_cf)

            self.model.train()
            return self.task.train_epoch(
                epoch=epoch,
                iters_per_epoch=self.iters_per_epoch,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                lr_scheduler=self.lr_scheduler,
                cuda_enabled=self.cuda_enabled,
                log_freq=self.log_freq,
                accum_grad_iters=self.accum_grad_iters,
            )

        # 第1阶段 
        elif self.config.model_cfg.finetune_stage == "1":
            self.model.train()
            return self.task.train_epoch(
                epoch=epoch,
                iters_per_epoch=self.iters_per_epoch,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                lr_scheduler=self.lr_scheduler,
                cuda_enabled=self.cuda_enabled,
                log_freq=self.log_freq,
                accum_grad_iters=self.accum_grad_iters,
            )  
            


    @torch.no_grad()
    def eval_epoch(self, valid_data_dir, use_cf=False, short_prompt=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        self.model.eval()
        ### (1) 创造dataset 和 dataloader
        eval_data_dir = valid_data_dir
        eval_dataset = ValidDataset(eval_data_dir, use_cf, short_prompt)
        if not use_cf:
            eval_dataloader = DataLoader(eval_dataset, batch_size=self.config.run_cfg.batch_size_eval, \
                                        shuffle=False, num_workers=self.config.run_cfg.num_workers, collate_fn=collate_fn_for_eval)
        else:
            eval_dataloader = DataLoader(eval_dataset, batch_size=self.config.run_cfg.batch_size_eval, \
                                        shuffle=False, num_workers=self.config.run_cfg.num_workers, collate_fn=collate_fn_for_eval_use_cf)
        ### (2) evaluation
        NDCG_10 = 0.0
        RECALL_10 = 0.0
        MRR = 0.0
        if not use_cf:
            for data in tqdm(eval_dataloader):
                descriptions, labels = data

                input_ids, input_embeds = self.prepare_lm_input(descriptions)
                with autocast():
                    features = self.model.chatglm2_model(
                        input_ids=input_ids,
                        inputs_embeds=input_embeds,
                        return_dict=True
                    )

                ##################### 获得item排序 #####################
                cos_sim = self.model.sim(features.unsqueeze(dim=1).to(self.model.I.device), self.model.I)
                _, top_indices = torch.topk(cos_sim, k=10, dim=1)
                _, all_indices = torch.topk(cos_sim, k=cos_sim.shape[1], dim=1)
                
                ##################### 更新指标 #####################
                for items, label, items_ in zip(top_indices, labels, all_indices):
                    items = items.tolist()
                    items_ = items_.tolist()
                    if label in items:
                        RECALL_10 += 1.0 / len(eval_dataset)
                        index = items.index(label)
                        NDCG_10 += (1.0 / math.log2(index + 2)) / len(eval_dataset)
                        MRR += (1.0 / (index + 1)) / len(eval_dataset)
                    else:
                        index = items_.index(label)
                        MRR += (1.0 / (index + 1)) / len(eval_dataset)

        else:
            expert_sample = {}
            for i in range(8):
                expert_sample[i] = []

            number = -1
            for data in tqdm(eval_dataloader):
                number += 1

                inputs, labels, input_embed, input_mask = data

                with autocast():
                    indices = self.model.obtain_features(
                        item_embs=input_embed, 
                        item_atts=input_mask, 
                        text=inputs
                    )
                
                for index, indice in enumerate(indices.tolist()):
                    expert_sample[indice].append(32 * number + index)


        return{"NDCG_10": NDCG_10, "RECALL_10": RECALL_10, "MRR": MRR}

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                sampler = None
                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=False,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )

            return loader

        loaders = []
        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False, stage_1=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        if stage_1:
            save_obj = {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "epoch": cur_epoch
            }
        else:
            save_obj = {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "epoch": cur_epoch,
                "I": self.model.I
            }
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    def translator_generate(self):
        model = self.unwrap_dist_model(self.model)
        model.eval()

        self.task.generate(
            iters_per_epoch=self.iters_per_epoch,
            model=model,
            data_loader=self.train_loader,
            scaler=self.scaler
        )

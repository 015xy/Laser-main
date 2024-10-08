"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
from datasets.datasets.base_dataset import BatchIterableDataset
import torch


class ArxivTextPairDataset(BatchIterableDataset):
    def __init__(self, cfg, mode):
        super(ArxivTextPairDataset, self).__init__(cfg, mode)
        self.max_length = cfg.arxiv_processor.train.max_length
        self.vocab_size = cfg.arxiv_processor.train.vocab_size
        self.cfg = cfg

    def _train_data_parser(self, data):

        # 处理训练数据
        if self.cfg.finetune_stage == "1":
            item_embedding = torch.tensor(eval(data[0][6]), dtype=torch.float32)
            mask = torch.tensor(eval(data[0][7]), dtype=torch.long)
            text = data[0][0]
            return item_embedding, mask, text
        
        else:  # 第2阶段
            if not self.cfg.use_cf:
                return data[0][0], data[0][1], data[0][2], data[0][3]
            else:
                input = data[0][0]
                label = data[0][1]
                negative = data[0][2]
                label_id = data[0][3]
                # input_id = data[0][4]
                # negative_id = data[0][5]
                input_embedding = data[0][6]
                input_mask = data[0][7]
                label_embedding = data[0][8]
                label_mask = data[0][9]
                negative_embedding = data[0][10]
                negative_mask = data[0][11]

                return input, label, negative, label_id, \
                torch.tensor(eval(input_embedding), dtype=torch.float32), torch.tensor(eval(input_mask), dtype=int), \
                torch.tensor(eval(label_embedding), dtype=torch.float32),  torch.tensor(eval(label_mask), dtype=int), \
                torch.tensor(eval(negative_embedding), dtype=torch.float32), torch.tensor(eval(negative_mask), dtype=int)
            

    def __len__(self):
        return self.row_count

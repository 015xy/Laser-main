"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from common.registry import registry
from models.base_model import all_gather_with_grad, concat_all_gather
from models.translator_models.translator import (
    TranslatorBase,
    compute_sim_matrix
)
from models.translator_models.translator_outputs import TranslatorOutput, TranslatorOutputFeatures

from transformers import BertTokenizer
from models.translator_models.Qformer import BertConfig, BertLMHeadModel


@registry.register_model("translator_arxiv")
class TranslatorQformerArxiv(TranslatorBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_arxiv": "train/pretrain_arxiv_stage1.yaml",
        "translator_generate_stage1": "train/pretrain_socialgenerate_stage1.yaml",
    }

    def __init__(
        self,
        alpha, 
        vision_hidden_state,
        top_k,
        num_experts,
        contrastive_loss,
        matching_loss,
        generative_loss,
        num_features=768,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32
    ):
        super().__init__()

        self.alpha = alpha

        self.tokenizer = self.init_tokenizer()

        self.top_k = top_k
        self.num_experts = num_experts

        self.contrastive_loss = contrastive_loss
        self.matching_loss = matching_loss
        self.generative_loss = generative_loss

        self.Qformer, self.query_experts = self.init_Qformer(
            num_experts, num_query_token, num_features, cross_attention_freq
        )

        # ------------------------
        self.image_proj = nn.Linear(vision_hidden_state, self.Qformer.config.hidden_size)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.behavior_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        self.router = nn.Linear(self.Qformer.config.hidden_size, num_experts)

    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("../bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def init_Qformer(cls, num_expert, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("../bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(encoder_config)
        checkpoint = torch.load("../bert-base-uncased/model.pth", map_location=lambda storage, loc: storage)

        Qformer.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        # --------------------
        query_experts = []
        for i in range(num_expert):
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            query_experts.append(query_tokens)
        return Qformer, nn.ParameterList(query_experts)

    def forward(self, samples):

        behavior_embeds = samples[0]
        mask = samples[1]  # [batch, num_vision_token] 
        text = samples[2]
        behavior_embeds = behavior_embeds.to(self.device)
        behavior_embeds = self.image_proj(behavior_embeds)
        # ---------------------

        behavior_atts = mask.to(self.device)

        score = nn.functional.softmax(self.router(behavior_embeds), dim=-1, dtype=torch.float32)  # (batch, items, num_expert)
        score = score.clone()
        score[mask == 0] = 0
        score = torch.mean(score, dim=1, keepdim=True)  # (batch, 1, num_expert)
        score, indices = torch.topk(score, k=self.top_k, dim=-1, largest=True)  # (batch, 1, top_k)
        score = nn.functional.softmax(score, dim=-1, dtype=torch.float32) 

        top_k_batch_query = []  
        for i in range(self.top_k):  
            batch_query = []  
            for j in range(score.shape[0]):  
                indice = indices[j, 0, i]
                batch_query.append(self.query_experts[indice])
            top_k_batch_query.append(torch.cat(batch_query, dim=0))

        output_batch_query = []  
        for batch_query in top_k_batch_query:
            output_batch_query.append(
                self.Qformer.bert(
                    query_embeds=batch_query,  # [batch, query_token, emb]
                    encoder_hidden_states=behavior_embeds,   # [batch, items, emb]
                    encoder_attention_mask=behavior_atts,    # [batch, items]
                    use_cache=True,
                    return_dict=True,
                )
            )

        behavior_feats = torch.zeros_like(self.behavior_proj(output_batch_query[0].last_hidden_state))  # (batch, query_token, embed_dim)

        for i in range(self.top_k):
            behavior_feats_temp = F.normalize(
                self.behavior_proj(output_batch_query[i].last_hidden_state), dim=-1
            )
            behavior_feats += torch.unsqueeze(score[:,:,i], -1) * behavior_feats_temp
        # ----------------------

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(behavior_embeds.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        loss_itc = None
        loss_itm = None
        loss_lm = None
        moe_loss = None

        ###============== Image-text Contrastive ===================###
        behavior_feats_all = concat_all_gather(
            behavior_feats
        )  # [batch_size*num_gpu, query_tokens, embed] 
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed]

        ###### 1. 计算i2t相似度 ######  [batch_size, batch_size*num_gpu]
        sim_q2t = torch.matmul(
            behavior_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp  # [batch_size, batch_size*num_gpu]

        ###### 2. 计算t2i相似度 ######  [batch_size, batch_size*num_gpu]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), behavior_feats_all.permute(0, 2, 1)
        ).squeeze()

        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        ###### 3. 得到target ######  [batch_size]
        rank = 0
        bs = behavior_embeds.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            behavior_embeds.device
        )  

        ###### 4. 计算对比学习的损失 ######
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        if self.contrastive_loss:
            loss = loss_itc

        if self.matching_loss:
            ###============== Image-text Matching ===================###
            text_input_ids_world = concat_all_gather(text_tokens.input_ids) # (batch*num_gpu, seq)
            text_attention_mask_world = concat_all_gather(text_tokens.attention_mask) # (batch*num_gpu, seq)
            behavior_embeds_world = all_gather_with_grad(behavior_embeds) # (batch*num_gpu, items, embed)
            behavior_atts_world = all_gather_with_grad(behavior_atts)

            ###### 1. 根据上面的sim_t2i与sim_i2t，选择负例 ###### 
            with torch.no_grad():
                weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
                weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
                weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

            # select a negative image for each text
            behavior_embeds_neg = []
            behavior_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                behavior_embeds_neg.append(behavior_embeds_world[neg_idx])
                behavior_atts_neg.append(behavior_atts_world[neg_idx])
            behavior_embeds_neg = torch.stack(behavior_embeds_neg, dim=0)  # [batch, vision_seq, embed]
            behavior_atts_neg = torch.stack(behavior_atts_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])
            text_ids_neg = torch.stack(text_ids_neg, dim=0)  # [batch, vision_seq, embed]
            text_atts_neg = torch.stack(text_atts_neg, dim=0)  # [batch, vision_seq, embed]

            # 将正例、负例做拼接
            text_ids_all = torch.cat(     # pos, pos, neg  [3 * batch, seq]
                [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
            ) 
            text_atts_all = torch.cat(    # pos, pos, neg  [3 * batch, seq]
                [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
                dim=0,
            )  
            
            # --------------------------
            behavior_embeds_all = torch.cat(                                        # [3 * batch, vision_seq, embed]
                    [behavior_embeds, behavior_embeds_neg, behavior_embeds], dim=0
            )  # pos, neg, pos  

            behavior_atts_all = torch.cat(                                        # [3 * batch, vision_seq]
                    [behavior_atts, behavior_atts_neg, behavior_atts], dim=0
            )

            score_for_match = nn.functional.softmax(self.router(behavior_embeds_all), dim=-1, dtype=torch.float32) # [3 * batch, vision_seq, num_expert]
            score_for_match = score_for_match.clone()
            score_for_match[behavior_atts_all == 0] = 0

            ######### 计算moe_loss
            # (1)计算p_expert
            p_expert = torch.sum(score_for_match, dim=[0,1]) / torch.sum(behavior_atts_all)
        
            # (2)计算f_expert
            f_expert = torch.zeros_like(p_expert)
            _, indices = torch.topk(score_for_match, k=self.top_k, dim=-1, largest=True)  # (batch, 1, k)
            unique_values, counts = torch.unique(indices, return_counts=True)
            for value, count in zip(unique_values, counts):
                f_expert[value] += count / torch.sum(behavior_atts_all)

            if self.num_experts > 1:
                moe_loss = self.alpha * f_expert.shape[0] * (p_expert * f_expert).sum()
                if self.contrastive_loss:
                    loss = loss + moe_loss
                else:
                    loss = moe_loss
            
            score_for_match = torch.mean(score_for_match, dim=1, keepdim=True)
            score_for_match, indices_for_match = torch.topk(score_for_match, k=self.top_k, dim=-1, largest=True)  # (batch, 1, k)
            score_for_match = nn.functional.softmax(score_for_match, dim=-1, dtype=torch.float32)
            
            ########

            query_tokens_list_for_match = [] 
            for i in range(self.top_k): 
                query_tokens = []   
                for j in range(score_for_match.shape[0]):  
                    indice = indices_for_match[j, 0, i]
                    query_tokens.append(self.query_experts[indice])
                query_tokens_list_for_match.append(torch.cat(query_tokens, dim=0))

            query_output_list_for_match = []
            for query_tokens in query_tokens_list_for_match:
                query_atts_itm = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                    behavior_embeds.device
                )
                attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)  # [3 * batch, query_token + seq]

                output_itm = self.Qformer.bert(
                    text_ids_all,                       
                    query_embeds=query_tokens,          
                    attention_mask=attention_mask_all,  
                    encoder_hidden_states=behavior_embeds_all,    
                    encoder_attention_mask=behavior_atts_all,   
                    return_dict=True,
                )

                query_output_list_for_match.append(output_itm.last_hidden_state[:, :query_tokens.size(1), :])  

            vl_embeddings = torch.zeros_like(query_output_list_for_match[0])  # [3 * batch, query_token, emb]

            for i in range(self.top_k):
            # for i in range(3):
                vl_embeddings_temp = query_output_list_for_match[i]
                vl_embeddings += torch.unsqueeze(score_for_match[:,:,i], -1) * vl_embeddings_temp  # [3 * batch, vision_seq, 1] * [3 * batch, query_token, emb]
            # -------------------------------
            
            vl_output = self.itm_head(vl_embeddings)  # [3 * batch, query_token, 2]
            logits = vl_output.mean(dim=1)            # [3 * batch, 2]

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(behavior_embeds.device)              # [3 * batch]
            loss_itm = F.cross_entropy(logits, itm_labels)

            if self.contrastive_loss:
                loss = loss + loss_itm
            else:
                loss = loss_itm

        if self.generative_loss:
            ##================= Image Captioning ========================##
            decoder_input_ids = text_tokens.input_ids.clone()   # [batch, seq]
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )

            query_atts = torch.ones(output_batch_query[0].last_hidden_state.size()[:-1], dtype=torch.long).to(   # [batch, query_token]
                self.device
            )  
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)   # [batch, query_token + seq]
            
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values_group=output_batch_query,
                return_dict=True,
                labels=labels,
                expert_score=score
            )

            loss_lm = lm_output.loss

            if self.contrastive_loss or self.matching_loss:
                loss = loss + loss_lm
            else:
                loss = loss_lm

        ##================= return loss ========================##
        return TranslatorOutput(
            loss=loss,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_moe=moe_loss
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=512,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        # behavior_embeds = torch.unsqueeze(samples[1], dim=1).to('cuda:1')

        # ------------------------
        behavior_embeds = torch.unsqueeze(samples[0], dim=1).to('cuda:1')

        behavior_embeds = self.image_proj(behavior_embeds)

        if not use_nucleus_sampling:
            behavior_embeds = behavior_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        behavior_atts = torch.ones(behavior_embeds.size()[:-1], dtype=torch.long).to(
            behavior_embeds.device
        )

        model_kwargs = {
            "encoder_hidden_states": behavior_embeds,
            "encoder_attention_mask": behavior_atts,
        }

        # -----------------------
        input_ids = (
            torch.LongTensor(samples[0].size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(behavior_embeds.device)
        )

        query_tokens = self.query_tokens.expand(behavior_embeds.shape[0], -1, -1).to(behavior_embeds.device)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):

        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return TranslatorOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        # vision_hidden_state
        vision_hidden_state = cfg.get("vision_hidden_state")

        # Behavior
        behavior_length = cfg.get("behavior_length", 384)

        # Text
        max_txt_len = cfg.get("max_txt_len", 32)

        # Q-Former
        alpha = cfg.get("alpha")
        top_k = cfg.get("top_k")
        num_expert = cfg.get("num_expert")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        # 预训练的三个任务
        contrastive_loss = cfg.get("contrastive_loss")
        matching_loss = cfg.get("matching_loss")
        generative_loss = cfg.get("generative_loss")

        model = cls(
            alpha=alpha,
            vision_hidden_state=vision_hidden_state,
            top_k=top_k,
            num_experts=num_expert,
            num_features=behavior_length,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            contrastive_loss=contrastive_loss,
            matching_loss=matching_loss,
            generative_loss=generative_loss,
        )

        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    def load_from_pretrained(self, url_or_filename):
        if url_or_filename:
            checkpoint = torch.load(url_or_filename, map_location=lambda storage, loc: storage)
            if "model_state_dict" in checkpoint.keys():
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            msg = self.load_state_dict(state_dict, strict=False)

            logging.info("load checkpoint from %s" % url_or_filename)

            return msg

        return

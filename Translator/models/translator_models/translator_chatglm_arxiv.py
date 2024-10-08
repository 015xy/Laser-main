"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from common.registry import registry
from models.translator_models.translator import TranslatorBase
from transformers import BertTokenizer
from models.translator_models.Qformer import BertConfig, BertLMHeadModel
from models.chatglm2 import ChatGLMForConditionalGeneration, ChatGLMTokenizer

from torch.cuda.amp import autocast

# IMAGE_TOKEN_ID = 101
PREFIX_TOKEN_ID = 101
SUFFIX_TOKEN_ID = 102

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


@registry.register_model("translator_arxiv_chatglm")
class TranslatorCHATGLMArxiv(TranslatorBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_arxiv": "train/pretrain_arxiv_stage2.yaml",
        "translator_generate_stage2": "train/pretrain_arxiv_generate_stage2.yaml"
    }

    def __init__(
        self,
        alpha, 
        vision_hidden_state,
        top_k,
        num_experts,
        config,
        num_features=768,
        num_query_token=32,
        chatglm2_model="",
        max_txt_len=2048,
        temp=0.05
    ):
        super().__init__()

        self.config = config
        
        if self.config.use_cf:
            # qformer
            self.bert_dir = config['bert_dir']
            self.tokenizer = self.init_tokenizer()
            self.Qformer, self.query_experts = self.init_Qformer(
                num_experts, num_query_token, num_features
            )
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
            self.Qformer.cls = None
            # moe
            self.alpha = alpha
            self.top_k = top_k
            self.num_experts = num_experts

            self.router = nn.Linear(self.Qformer.config.hidden_size, num_experts)
            self.image_proj = nn.Linear(vision_hidden_state, self.Qformer.config.hidden_size)
            
        self.llm_dir = config['llm_dir']
        self.chatglm2_tokenizer = ChatGLMTokenizer.from_pretrained(self.llm_dir, use_fast=False, trust_remote_code=True)
        self.chatglm2_model = ChatGLMForConditionalGeneration.from_pretrained(self.llm_dir)
        for _, param in self.chatglm2_model.named_parameters():
            param.requires_grad = False
        
        if self.config.use_cf:
            self.chatglm2_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.chatglm2_model.config.hidden_size
            )

        self.max_txt_len = max_txt_len
        self.sim = Similarity(temp)
        self.loss_fct = nn.CrossEntropyLoss()

        self.suffix = nn.Parameter(
            # batch, token, emb
            torch.zeros(1, 1, 4096)
        )
        self.suffix.data.normal_(mean=0.0, std=0.02)

        if self.config.use_prefix and not self.config.use_cf:
            self.prefix = nn.Parameter(
            # batch, token, emb
            torch.zeros(1, self.config.num_query_token, 4096)
            )
            self.prefix.data.normal_(mean=0.0, std=0.02)
        
        # self.head_project = nn.Sequential(
        #     nn.Linear(self.chatglm2_model.config.hidden_size, int(self.chatglm2_model.config.hidden_size / 4)),
        #     nn.GELU(),
        #     nn.Linear(int(self.chatglm2_model.config.hidden_size / 4), self.chatglm2_model.config.hidden_size)
        # )
        
    def init_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.bert_dir)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def init_Qformer(self, num_expert, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(self.bert_dir)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        Qformer = BertLMHeadModel(encoder_config)

        if self.config.use_bert_pretrained:
            checkpoint = torch.load(self.bert_dir+"/model.pth", map_location=lambda storage, loc: storage)
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

    def prepare_lm_input(self, texts: List[str], prefix=None, short_prompt=False):
        
        # 获得input_ids
        input_ids = self.chatglm2_tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt').input_ids.to(self.device)
        
        # 获得input_embs
        input_embs = self.chatglm2_model.transformer.embedding.word_embeddings(input_ids)

        batch_size = len(texts)
        suffix = self.suffix.expand(batch_size, -1, -1)

        input_embs = torch.cat([input_embs, suffix], dim=1)

        if prefix is not None:
            input_embs = torch.cat([prefix, input_embs], dim=1)
        elif self.config.use_prefix:
            prefix = self.prefix.expand(batch_size, -1, -1)
            input_embs = torch.cat([prefix, input_embs], dim=1)
        
        # 更新input_ids
        suffix_ids = torch.tensor([SUFFIX_TOKEN_ID] * batch_size).view(batch_size, 1).to(self.device)
        input_ids = torch.cat([input_ids, suffix_ids], dim=1)

        if prefix is not None:
            prefix_ids = torch.tensor([PREFIX_TOKEN_ID] * batch_size * self.config.num_query_token)\
                         .view(batch_size, self.config.num_query_token).to(self.device)
            input_ids = torch.cat([prefix_ids, input_ids], dim=1)

        # 返回input_ids和input_embs
        return input_ids, input_embs.transpose(0, 1).contiguous()

    def obtain_features(self, item_embs, item_atts, text):
        # （1）处理input数据
        multimodal_embeds = item_embs.to(self.device)  
        multimodal_embeds = self.image_proj(multimodal_embeds) 
        multimodal_atts = item_atts.to(self.device)     

        score = nn.functional.softmax(self.router(multimodal_embeds), dim=-1, dtype=torch.float32)  # [batch, seq, expert]
        score = score.clone()
        score[multimodal_atts == 0] = 0

        # score = nn.functional.softmax(self.router(multimodal_embeds), dim=-1, dtype=torch.float32)  # [batch, seq, expert]
        score = torch.mean(score, dim=1) # [batch, expert]
        indices = torch.argmax(score, dim=1)

        return indices

        
    def forward(self, samples):
             
        if not self.config.use_cf:
            if self.config.finetune_stage == "2-1":
                input_text = samples[0]  
                label_text = samples[1]
                negative_label_text = samples[2]  # batch

                # with torch.no_grad():
                # 处理label_text
                positive_ids, positive_embeds = self.prepare_lm_input(label_text, short_prompt=self.config.short_prompt)
                positive_features = self.chatglm2_model(
                    input_ids=positive_ids,
                    inputs_embeds=positive_embeds,
                    return_dict=True
                )

                # 处理negative_label_text
                negative_ls = []
                for negative in negative_label_text:
                    negative_ids, negative_embeds = self.prepare_lm_input(eval(negative), short_prompt=self.config.short_prompt)
                    negative_output = self.chatglm2_model(
                        input_ids=negative_ids,
                        inputs_embeds=negative_embeds,
                        return_dict=True
                    )
                    negative_ls.append(negative_output)
                negative_features = torch.stack(negative_ls, dim=0)

                # 拼接positive_features 与 negative_features
                label_features = torch.cat([positive_features.unsqueeze(dim=1), negative_features], dim=1)

                # 处理input_text
                input_ids, input_embeds = self.prepare_lm_input(input_text, short_prompt=self.config.short_prompt)
                output_features = self.chatglm2_model(
                    input_ids=input_ids,
                    inputs_embeds=input_embeds,
                    return_dict=True
                )

                # 获得相似度评分
                cos_sim = self.sim(output_features.unsqueeze(dim=1), label_features)

                # 计算loss
                label = torch.zeros(len(input_text), dtype=int).to(self.device)
                loss = self.loss_fct(cos_sim, label)
                return {"loss": loss}
        
            elif self.config.finetune_stage in ["2-2", "2-3"]:
                input_text = samples[0]  

                # 处理input_text
                input_ids, input_embeds = self.prepare_lm_input(input_text, short_prompt=self.config.short_prompt)
                output_features = self.chatglm2_model(
                    input_ids=input_ids,
                    inputs_embeds=input_embeds,
                    return_dict=True
                )

                # 获得相似度评分
                cos_sim = self.sim(output_features.unsqueeze(dim=1).to(self.I.device), self.I)

                # 计算loss
                label = samples[3].to(cos_sim.device)
                loss = self.loss_fct(cos_sim, label)
                return {"loss": loss}
            
        else:
            input = samples[0]   # List[str]
            input_embedding = samples[4]  # torch.Size([batch, input_item_num, item_emb])
            input_mask = samples[5]  # torch.Size([batch, input_item_num])
            
            label = samples[1]   # List[str]
            label_embedding = samples[6]  # torch.Size([batch, 1, item_emb])
            label_mask = samples[7]  # torch.Size([batch, 1])
            
            negative = [eval(d) for d in samples[2]]  # List[List[str]]
            negative_embedding = samples[8]  # torch.Size([batch, neg_item_num, item_emb])
            negative_mask = samples[9]  # torch.Size([batch, neg_item_num])
            
            label_id = samples[3].to(self.config.I_device)  # torch.Size([batch])
            
            # 获得input的feature
            input_features, input_moe_loss = self.obtain_features(
                item_embs=input_embedding, 
                item_atts=input_mask, 
                text=input
            )
            moe_loss = input_moe_loss

            # 如果是第一阶段，需要分别得到正负label的feature
            if self.config.finetune_stage == "2-1":
                
                # 读取label的feature
                label_features, label_moe_loss = self.obtain_features(
                    item_embs=label_embedding, 
                    item_atts=label_mask, 
                    text=label
                )
                moe_loss += label_moe_loss
                
                # 读取negative的feature
                negative_features = []
                for i in range(len(negative)):
                    negative_feature, negative_moe_loss = self.obtain_features(
                        item_embs=negative_embedding[i].unsqueeze(dim=1), 
                        item_atts=negative_mask[i].unsqueeze(dim=1), 
                        text=negative[i]
                    )
                    negative_features.append(negative_feature)
                    moe_loss += negative_moe_loss
                if len(negative_features) > 1:
                    negative_features = torch.stack(negative_features, dim=0)
                else:
                    negative_features = negative_features[0].unsqueeze(dim=0)

                self.I = torch.cat([label_features.unsqueeze(dim=1), negative_features], dim=1).to(self.config.I_device)

            # 获得相似度评分
            cos_sim = self.sim(input_features.unsqueeze(dim=1).to(self.I.device), self.I)

            # 计算loss
            if self.config.finetune_stage == "2-1":
                label_id = torch.zeros(len(cos_sim), dtype=int).to(self.I.device)
            loss = self.loss_fct(cos_sim, label_id)

            if self.config.num_expert > 1:
                loss += self.alpha * moe_loss.to(self.I.device)

            return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        prompts=[],
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=2048,
        min_length=1,
        top_p=0.8,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.65,
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
        device = self.Qformer.bert.device
        multimodal_embeds = samples[1].unsqueeze(dim=1).to(device)
        bs = len(samples[0])
        title = samples[3]
        instruction = [prompts[0]] * bs

        question_prompt_pre = prompts[1]

        categories = prompts[2]

        question_prompt = prompts[3]
        qustion = question_prompt_pre + categories + question_prompt

        with self.maybe_autocast():
            multimodal_atts = torch.ones(multimodal_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1)
            text_Qformer = self.tokenizer(
                instruction,
                padding='max_length',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=multimodal_embeds,
                encoder_attention_mask=multimodal_atts,
                return_dict=True,
            )

            vtokens = self.chatglm2_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])

            input_ids, labels, inputs_embeds = self.prepare_lm_input(
                vtokens=vtokens, text_input=instruction, answer=None
            )

            from transformers.generation.utils import LogitsProcessorList
            logits_processor = LogitsProcessorList()
            from models.chatglm2.modeling_chatglm import InvalidScoreLogitsProcessor
            logits_processor.append(InvalidScoreLogitsProcessor())

            gen_kwargs = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": 1,
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "logits_processor": logits_processor
            }

            outputs = self.chatglm2_model.generate(input_ids, inputs_embeds=inputs_embeds, **gen_kwargs)

            response_output = []
            for i in range(multimodal_embeds.shape[0]):
                outputs_i = outputs.tolist()[i][len(input_ids[i]):]
                response0 = self.chatglm2_tokenizer.decode(outputs_i)
                response0 = self.chatglm2_model.process_response(response0)

                #先总结再回答
                if len(response0) > max_length - len(qustion) - 1:
                    response0 = response0[:max_length - len(qustion) - 1]
                summary_prompt = prompts[4].format(title, response0, qustion)
                gen_kwargs = {
                    "max_length": max_length,
                    "min_length": 100
                }
                response2, history = self.chatglm2_model.chat(tokenizer=self.chatglm2_tokenizer,
                                                            query=summary_prompt,
                                                            **gen_kwargs)
                response_output.append(response2)

            return response_output

    @classmethod
    def from_config(cls, cfg):
        # multimodal
        vision_hidden_state = cfg.get("vision_hidden_state")
        num_features = cfg.get("num_features", 768)
        # Text
        max_txt_len = cfg.get("max_txt_len", 32)
        # Q-Former
        alpha = cfg.get("alpha")
        top_k = cfg.get("top_k")
        num_expert = cfg.get("num_expert")
        num_query_token = cfg.get("num_query_token")
        # similarity
        temp = cfg.get("temp")

        model = cls(
            config=cfg,
            alpha = alpha,
            vision_hidden_state=vision_hidden_state,
            top_k = top_k,
            num_experts=num_expert,
            num_features=num_features,
            num_query_token=num_query_token,
            max_txt_len=max_txt_len,
            temp=temp,
        )
        model.load_checkpoint_from_config(cfg)

        return model

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

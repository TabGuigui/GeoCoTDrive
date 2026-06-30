# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

# from ..vlm_utils.custom_qwen import CustomQwen2_5_VLForConditionalGeneration
from ..vlm_utils.custom_llava import CustomLlavaForConditionalGeneration

def make_row_mask_hook(allowed_row_indices):
    # allowed in training, form like [new_id]
    allowed = torch.tensor(allowed_row_indices, dtype=torch.long)
    def hook(grad: torch.Tensor):
        if grad is None:
            return None
    
        # grad has the same size as weights [vocab_size, hidden_size]
        grad_allowed = grad[allowed]
        grad *= 0
        grad[allowed] = grad_allowed
        return grad
    return hook

def load_model(base_model,tokenizer_path, use_lora, frozen, llm_lora_rank=16):

    if 'Qwen2.5' in base_model:
        model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map= {'':torch.cuda.current_device()}, local_files_only=True, 
        )
        model.gradient_checkpointing_enable()
    
    
        if use_lora:
            print('Using LoRA with rank', llm_lora_rank)
            peft_config = LoraConfig(
                    r=llm_lora_rank,
                    lora_alpha=16,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM")
            model = get_peft_model(model, peft_config)

            for name, param in model.named_parameters():
                if 'coords' in name:
                    param.requires_grad = True

        finetune_new_token = 'with-new-special-tokens' in base_model and use_lora
        if finetune_new_token:
            new_lm_head = None
            new_embed_tokens = None
            for name, param in model.named_parameters():
                print('loading', name,)
                if '.lm_head' in name or 'embed_tokens' in name:
                    print('loading selected', name, param.shape)

                    if '.lm_head' in name:
                        new_lm_head = nn.Linear(param.shape[1], 2, bias=False).to(param.device)
                        new_lm_head.weight.data = param.data[-2:, :]
                        new_lm_head.requires_grad = True
                    elif '.embed_tokens' in name:
                        new_embed_tokens = nn.Embedding(2, param.shape[1]).to(param.device)
                        new_embed_tokens.weight.data = param.data[-2:, :]
                        new_embed_tokens.requires_grad = True
                        # for tied weights
                        if model._tied_weights_keys is not None:
                            # tried to create new_lm_head
                            new_lm_head = nn.Linear(param.shape[1], 2, bias=False).to(param.device)
                            new_lm_head.weight.data = param.data[-2:, :]
                            new_lm_head.requires_grad = True

            if new_lm_head is not None:
                model.base_model.model.new_lm_head = new_lm_head
            if new_embed_tokens is not None:
                model.base_model.model.model.language_model.new_embed_tokens = new_embed_tokens

    elif 'llava' in base_model:
        model = CustomLlavaForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map={'':torch.cuda.current_device()}, local_files_only=True, 
        )
        model.gradient_checkpointing_enable()
    
        if use_lora:
            peft_config = LoraConfig(
                    r=llm_lora_rank,
                    lora_alpha=llm_lora_rank,
                    target_modules=("k_proj", "v_proj", "o_proj"),
                    exclude_modules='.*vision_tower.*',
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM")
            model = get_peft_model(model, peft_config)

            for name, param in model.named_parameters():
                if 'coords' in name:
                    param.requires_grad = True

        finetune_new_token = 'with-new-special-tokens' in base_model and use_lora
        if finetune_new_token:
            for name, param in model.named_parameters():
                if '.lm_head' in name or 'embed_tokens' in name:
                    param.requires_grad = True
                    row_mask_hook = make_row_mask_hook([32002]) #  geocot waypoint
                    param.register_hook(row_mask_hook)

    for param in filter(lambda p: p.requires_grad,model.parameters()):
        param.data = param.data.to(torch.float32)


    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    # clear memory
    torch.cuda.empty_cache()
               

    return model

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        
        locations = locations.reshape(h, w, 2)
        
        return locations
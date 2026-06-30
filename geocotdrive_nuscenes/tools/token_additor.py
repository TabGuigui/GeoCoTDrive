# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------


import transformers
import torch

# Monkey patch to move torch.distributed._tensor.DTensor to torch.distributed.tensor.DTensor
# This is used in save_pretrained in transformers 4.32.1
from transformers.pytorch_utils import is_torch_greater_or_equal, is_torch_xla_available, storage_ptr, storage_size
def patched_id_tensor_storage(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    """
    if is_torch_greater_or_equal("2.1", accept_dev=True):
        from torch.distributed._tensor import DTensor

        if isinstance(tensor, DTensor):
            local_tensor = tensor.to_local()
            return tensor.device, local_tensor.storage().data_ptr(), tensor.nbytes

    if tensor.device.type == "xla" and is_torch_xla_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)
transformers.pytorch_utils.id_tensor_storage = patched_id_tensor_storage
from torch.distributed._tensor import DTensor
transformers.modeling_utils.DTensor = DTensor


from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
def add_token(local_ckpt_path, extra_tokens, new_path):
    if 'llava' in local_ckpt_path:
        tokenizer = AutoTokenizer.from_pretrained(local_ckpt_path,  local_files_only=True)
        processor = AutoProcessor.from_pretrained(local_ckpt_path,  local_files_only=True)
        model = LlavaForConditionalGeneration.from_pretrained(local_ckpt_path, torch_dtype=torch.bfloat16,device_map={'':torch.cuda.current_device()},  local_files_only=True)
    elif 'Qwen' in local_ckpt_path:
        tokenizer = AutoTokenizer.from_pretrained(local_ckpt_path,  local_files_only=True)
        processor = AutoProcessor.from_pretrained(local_ckpt_path,  local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_ckpt_path, torch_dtype=torch.bfloat16,device_map={'':torch.cuda.current_device()},  local_files_only=True)
    else:
        raise NotImplementedError(f"Model type not supported for path {local_ckpt_path}")
    
    # add special tokens
    special_token_dict = {'additional_special_tokens': tokenizer.additional_special_tokens + extra_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_token_dict)
    print("We have added", num_added_toks, "tokens")

    # resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # use the average of existing token embeddings to initialize the new token embeddings
    with torch.no_grad():
        for token in extra_tokens:
            new_token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"Setting embedding for new token: {token} (ID: {new_token_id})")
            avg_embedding = model.get_input_embeddings().weight.mean(dim=0)
            model.get_input_embeddings().weight[new_token_id] = avg_embedding
    
    # print all special tokens
    print("All special tokens:", tokenizer.all_special_tokens)

    print(f"New token ID: {new_token_id}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Model embedding layer size: {model.get_input_embeddings().weight.size(0)}")

    model.config.vocab_size = len(tokenizer)

    # save the new model and tokenizer
    model.save_pretrained(new_path)
    processor.save_pretrained(new_path)
    tokenizer.save_pretrained(new_path)
    print(f"Model, tokenizer, and processor with new tokens saved to {new_path}")

if __name__ == "__main__":
    local_ckpt_path = "../ckpts/llava-1.5-7b-hf"  # e.g., ./ckpts/llava-1.5-7b-hf"  "path_to_your_model_checkpoint"
    new_path = "../ckpts/llava-1.5-7b-hf-with-new-special-tokens-waypoint"  # e.g., "./ckpts/llava-1.5-7b-hf-with-new-special-tokens"  "path_to_save_new_model_with_added_tokens"
    extra_tokens = ['<GEO_COT>', "<WAYPOINT>"]


    add_token(local_ckpt_path, extra_tokens, new_path)
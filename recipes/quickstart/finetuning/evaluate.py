import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_uspto_dataset,
)
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
##load model
train_config= TRAIN_CONFIG()

model_id="/afs/crc.nd.edu/user/x/xhuang2/llama_test/llama-recipes/recipes/finetuning/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_id)

model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
##load lora
peft_model_id = '/afs/crc.nd.edu/user/x/xhuang2/llama_test/llama-recipes/recipes/finetuning/PEFT_e200'
model.load_adapter(peft_model_id)

##load data of uspto
dataset_val =  get_uspto_dataset(None ,tokenizer, "test")
val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
def evaluation(model, eval_dataloader, tokenizer):
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0
    for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    
    return eval_epoch_loss, eval_preds

def inference(model, eval_dataloader, tokenizer):
    model.eval()
    infer = []



eval_loss, eval_preds = evaluation(model, eval_dataloader, tokenizer)
import pandas as pd

df = pd.read_csv('./datasets/combined_finetuning_data_test.csv')
df['extracted'] = eval_preds
df.to_csv('./output_finetune_llama_7b_e200.csv')
import numpy as np 
np.savetxt('output_finetune_llama_7b_e200.txt',eval_preds, fmt='%s')


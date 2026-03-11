import math
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torch.optim import AdamW 
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    PrefixTuningConfig, 
    TaskType, 
    set_peft_model_state_dict
)
import constants
import parser
import process  
import gc

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Multi-process environment not detected, running in single GPU mode")
        return -1, -1, -1
    
    torch.cuda.set_device(local_rank)
    print(f'| Distributed initialization (Rank {rank}, Local Rank {local_rank})', flush=True)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    dist.barrier()
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0 or rank == -1
    
    args = parser.get_parser()
    
    if rank != -1:
        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'
    else:
        args.device = 'cuda:0'

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Memory fragmentation optimization
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    dir_models = os.path.join(args.dir_models_tuned)
    
    if is_main_process:
        dir_tb_log = os.path.join(dir_models, 'logs')
        os.makedirs(dir_tb_log, exist_ok=True)
        writer = SummaryWriter(dir_tb_log)
        print(f"Training started! Case: {args.case_id}, Model: {args.model}")
    else:
        writer = None
    
    # === LoRA configuration (full module optimization for Qwen/Llama3) ===
    if constants.cases[args.case_id]['method'] == 'lora':
        # For modern models (Qwen or Llama3), fine-tuning all linear layers yields the best results
        if any(m in args.model.lower() for m in ["llama3", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "v_proj"] 

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r=constants.LORA_R, 
            lora_alpha=constants.LORA_ALPHA, 
            lora_dropout=constants.LORA_DROPOUT,
            target_modules=target_modules
        )
    elif constants.cases[args.case_id]['method'] == 'prefix_tuning':
        config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=constants.PROMPT_LEN)
    
    model_path = constants.MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Unified CausalLM family loading logic (Llama/Qwen/Falcon)
    if any(m in args.model.lower() for m in ["llama", "qwen", "falcon"]):
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }
        
        if rank != -1:
            load_kwargs["device_map"] = {"": local_rank}
        else:
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        # Padding token handling: resolve Qwen's missing pad_token error
        if tokenizer.pad_token is None:
            if "qwen" in args.model.lower():
                tokenizer.pad_token = ""
            else:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        # Training typically uses right padding
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        # Seq2Seq (T5 family models)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map={"": local_rank} if rank != -1 else "auto")

    # Apply PEFT fine-tuning configuration
    model = get_peft_model(model, config)

    # Resume from checkpoint
    if args.resume_checkpoint is not None:
        ckpt_path = os.path.join(args.resume_checkpoint, "adapter_model.safetensors")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.resume_checkpoint, "adapter_model.bin")
        if is_main_process:
            print(f"--> Resuming weights from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device), strict=False)
            
    # Distributed DDP wrapping
    if rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process:
        (model.module if hasattr(model, 'module') else model).print_trainable_parameters()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    trn_dataset = process.load_data(args, task='trn')
    val_dataset = process.load_data(args, task='val')
    
    if rank != -1:
        trn_sampler = DistributedSampler(trn_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        trn_loader = process.get_loader_distributed(trn_dataset, tokenizer, args.batch_size, args.model, trn_sampler)
        val_loader = process.get_loader_distributed(val_dataset, tokenizer, args.batch_size, args.model, val_sampler)
    else:
        trn_loader = process.get_loader(trn_dataset, tokenizer, args.batch_size, args.model)
        val_loader = process.get_loader(val_dataset, tokenizer, args.batch_size, args.model)
    
    num_training_steps = len(trn_loader) * args.trn_epochs
    optimizer, lr_scheduler = define_optimizer(model, num_training_steps, args)

    best_val_loss = math.inf
    n_steps = 0
    
    # Training loop
    for epoch in range(args.trn_epochs):
        if rank != -1: trn_sampler.set_epoch(epoch)
        model.train()
        if is_main_process: pbar = tqdm(total=len(trn_loader), desc=f"Epoch {epoch}")
        
        epoch_losses = []
        for batch_idx, batch in enumerate(trn_loader):
            n_steps += 1
            batch.pop('idx', None)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if (n_steps % args.grad_accum_steps == 0) or (batch_idx == len(trn_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            detached_loss = outputs.loss.detach().float()
            epoch_losses.append(detached_loss)
            
            if is_main_process:
                if writer: writer.add_scalar('trn_loss', detached_loss, n_steps)
                pbar.update(1)
                if n_steps % 10 == 0: pbar.set_postfix({'loss': f"{detached_loss.item():.4f}"})

        if is_main_process: pbar.close()

        # Validation set evaluation
        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch.pop('idx', None)
                batch = {k: v.to(args.device) for k, v in batch.items()}
                val_losses.append(model(**batch).loss.detach().float())

        # Synchronize losses across GPUs
        if rank != -1:
            avg_trn = torch.stack(epoch_losses).mean().to(args.device)
            avg_val = torch.stack(val_losses).mean().to(args.device)
            dist.all_reduce(avg_trn, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_val, op=dist.ReduceOp.SUM)
            trn_loss_epoch, val_loss_epoch = avg_trn/world_size, avg_val/world_size
        else:
            trn_loss_epoch = sum(epoch_losses) / len(epoch_losses)
            val_loss_epoch = sum(val_losses) / len(val_losses)

        if is_main_process:
            print(f"Epoch {epoch}: trn_loss={trn_loss_epoch:.4f}, val_loss={val_loss_epoch:.4f}")
            actual_model = model.module if hasattr(model, 'module') else model
            # Save current epoch model
            actual_model.save_pretrained(os.path.join(dir_models, f'{epoch}'))
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                print(f"Found a better model! Validation loss: {best_val_loss:.4f}")

    cleanup_distributed()
    
def define_optimizer(model, num_training_steps, args):
    case = constants.cases[args.case_id]
    optimizer = AdamW(model.parameters(), lr=case['lr0'])
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=args.lr_n_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, lr_scheduler

if __name__ == '__main__':
    main()
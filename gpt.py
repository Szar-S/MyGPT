import os
import torch
import torch.nn as nn
import torch.distributed as dist
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
import sys
import textwrap
import glob
import re
from config import config
from refresh import extract_text_from_pdfs_and_txts, create_tokenizer, token_save

# =====================
# 1. TOKENIZER SETUP (Improved)
# =====================
def initialize_tokenizer(forModel=None):
    if forModel is None:
        forModel = config["forModel"]
        
    tokenizer_path = os.path.join(forModel, config["bpe_tokenizer"])
    os.makedirs(forModel, exist_ok=True)
    data_path = os.path.join(forModel, config["data_corpus"])
    
    # Create tokenizer if missing or invalid
    if not os.path.exists(tokenizer_path) or os.path.getsize(tokenizer_path) == 0:
        if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:    
            text = extract_text_from_pdfs_and_txts()
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
        tokenizer = create_tokenizer(text)
    else:
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            # Validate tokenizer
            if tokenizer.get_vocab_size() < 100:
                raise ValueError("Invalid vocabulary size")
        except Exception as e:
            print(f"Tokenizer error: {e}. Rebuilding...")
            if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:    
                text = extract_text_from_pdfs_and_txts()
            else:
                with open(data_path, "r", encoding="utf-8") as f:
                    text = f.read()
            tokenizer = create_tokenizer(text)
    
    # Add special tokens if missing
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    for token in special_tokens:
        if tokenizer.token_to_id(token) is None:
            tokenizer.add_special_tokens([token])
    
    return tokenizer

# =====================
# 2. DATASET HANDLING (Optimized)
# =====================
class TextDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len=None, split='train'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len or config["seq_len"]
        self.corpus_path = os.path.join(config["forModel"], config["data_corpus"])
        
        self.tokens = token_save(tokenizer, corpus_path)
        self.num_tokens = len(self.tokens)
        
        # Split dataset
        split_idx = int(0.9 * self.num_tokens)
        if split == 'train':
            self.tokens = self.tokens[:split_idx]
        else:  # 'val'
            self.tokens = self.tokens[split_idx:]
        self.num_tokens = len(self.tokens)
        
        self.special_ids = {
            "bos": tokenizer.token_to_id("<bos>"),
            "eos": tokenizer.token_to_id("<eos>"),
            "pad": tokenizer.token_to_id("<pad>")
        }
        
        self.bos_id = self.special_ids["bos"]
        self.eos_id = self.special_ids["eos"]
        self.pad_id = self.special_ids["pad"]

    def __len__(self):
        return max(1, (self.num_tokens - 1) // self.seq_len)

    def __getitem__(self, idx):
        start_pos = idx * self.seq_len
        end_pos = start_pos + self.seq_len + 1
        
        if end_pos > self.num_tokens:
            # Calculate actual available tokens
            available = self.num_tokens - start_pos
            # Get available tokens
            chunk = list(self.tokens[start_pos:start_pos + available])
            # Pad to full length +1 (for labels)
            pad_len = (self.seq_len + 1) - available
            chunk += [self.pad_id] * pad_len
            # Set last real token to <eos>
            if available > 0:
                chunk[available - 1] = self.eos_id
        else:
            chunk = list(self.tokens[start_pos:end_pos])
        return {
            "input_ids": torch.tensor(chunk[:self.seq_len], dtype=torch.long),
            "labels": torch.tensor(chunk[1:self.seq_len+1], dtype=torch.long)
        }

# =====================
# 3. MODEL DEFINITION (Enhanced)
# =====================
class NanoGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=None, n_layers=None, n_heads=None):
        super().__init__()
        # Configurable parameters with defaults
        embed_size = embed_size or config["embed_size"]
        n_layers = n_layers or config["n_layers"]
        n_heads = n_heads or config["n_heads"]
        dropout_rate = config.get("dropout_rate", 0.1)
        
        # Embedding layers
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(config["seq_len"], embed_size)
        self.embed_norm = nn.LayerNorm(embed_size)  # Pre-normalization
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=4*embed_size,
                dropout=dropout_rate,
                activation='gelu',  # Better activation
                batch_first=True,
                norm_first=True  # Pre-LayerNorm architecture
            ) for _ in range(n_layers)
        ])
        
        # Output layers
        self.ln_final = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Position embeddings
        pos_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
        
        # Embedding combination
        token_embeds = self.token_embed(input_ids)
        pos_embeds = self.pos_embed(pos_ids)
        x = token_embeds + pos_embeds
        x = self.embed_norm(x)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'))
        mask = mask.masked_fill(mask == 0, 0).to(device)
        
        # Transformer processing
        for block in self.blocks:
            x = block(x, src_mask=mask)
        
        # Final output
        x = self.ln_final(x)
        return self.lm_head(x)

# =====================
# 4. TRAINING FUNCTION (Optimized)
# =====================
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def evaluate(model, val_loader, loss_fn, device, rank):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda', enabled=config["use_amp"]):
                logits = model(inputs)
                # Calculate loss per token (ignore padding)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    # Gather metrics across devices if DDP
    if dist.is_initialized():
        total_loss_tensor = torch.tensor(total_loss).to(device)
        total_tokens_tensor = torch.tensor(total_tokens).to(device)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = total_loss_tensor.item()
        total_tokens = total_tokens_tensor.item()
    
    avg_loss = total_loss / total_tokens
    # FIX: Proper perplexity calculation
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 15 else float('inf')
    
    if rank == 0:
        print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

def train_model(rank, tokenizer, world_size=0, forModel=config["forModel"], model=None):
    # DDP setup
    use_ddp = config["use_ddp"] and world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size)
        device = rank
        torch.cuda.set_device(device)
    else:
        device = torch.device(config["device"])
    
    # Path setup
    corpus_path = os.path.join(forModel, config["data_corpus"])
    save_path = os.path.join(forModel, config["model_path"])
    checkpoint_path = os.path.join(forModel, "checkpoint.pth")
    
    # Initialize datasets
    if not os.path.exists(corpus_path) or os.path.getsize(corpus_path) == 0:
        extract_text_from_pdfs_and_txts()
    
    train_dataset = TextDataset(corpus_path, tokenizer, config["seq_len"], split='train')
    val_dataset = TextDataset(corpus_path, tokenizer, config["seq_len"], split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    # Model initialization
    vocab_size = tokenizer.get_vocab_size()
    if model is None:
        model = NanoGPT(vocab_size=vocab_size)
        # Load existing model weights if available
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            model.load_state_dict(torch.load(save_path, map_location=device))
            if rank == 0:
                print(f"Loaded existing model weights from {save_path}")
    
    # Load checkpoint if available
    start_epoch = 0
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if rank == 0:
            print(f"Resuming from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1
    
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])
    
    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.98),
        eps=1e-6
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    scaler = torch.amp.GradScaler('cuda', enabled=config["use_amp"])
    
    # Training loopbest_val_loss = float('inf')
    for epoch in range(start_epoch, config["epochs"]):
        # Create data loader
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        ) if use_ddp else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=config["num_workers"],
            persistent_workers=True
        )
        
        if sampler:
            sampler.set_epoch(epoch)
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * config["epochs"],
            eta_min=config["learning_rate"] * 0.1
        )
        
        # Training
        model.train()
        total_loss = 0.0
        total_steps = 0
        optimizer.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", disable=(rank != 0))
        
        for step, batch in enumerate(progress):
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda', enabled=config["use_amp"]):
                logits = model(inputs)
                # Calculate actual loss
                actual_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                # Scale loss for gradient accumulation
                scaled_loss = actual_loss / config["gradient_accumulation_steps"]
            
            scaler.scale(scaled_loss).backward()
            
            # Accumulate actual loss
            total_loss += actual_loss.item()
            total_steps += 1
            
            # Display actual loss in progress bar
            progress.set_postfix(loss=f"{actual_loss.item():.4f}")
            
            # Gradient accumulation
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        
        # Calculate epoch loss
        epoch_loss = total_loss / total_steps
        if rank == 0:
            print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device, rank)
        
        # Save best model
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"Saved best model (loss={val_loss:.4f}, ppl={val_ppl:.2f})")
        
        # Save checkpoint
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict() if use_ddp else model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
    
    # Delete checkpoint after training
    if rank == 0 and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Deleted checkpoint: {checkpoint_path}")
    
    # Cleanup
    if use_ddp:
        dist.destroy_process_group()
    return model

# =====================
# 5. GENERATION (Enhanced)
# =====================
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    # Apply nucleus sampling
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter indices
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits

def apply_repetition_penalty(logits, input_ids, penalty=1.5, depth=20):  # Increased penalty and depth
    if len(input_ids) == 0:
        return logits
        
    # Consider last N tokens
    recent_ids = input_ids[-depth:] if len(input_ids) > depth else input_ids
    unique_ids = torch.unique(recent_ids)
    
    for uid in unique_ids:
        if logits[uid] > 0:
            # Apply penalty based on frequency
            count = (recent_ids == uid).sum().item()
            logits[uid] /= penalty * (1 + count * 0.2)
            
    return logits

def generate_text(model, tokenizer, prompt, max_length=150, temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    
    # Handle empty prompt
    if not input_ids:
        input_ids = [tokenizer.token_to_id("<bos>")]
    
    # Truncate to model capacity
    max_ctx = config["seq_len"] - max_length
    if len(input_ids) > max_ctx:
        input_ids = input_ids[-max_ctx:]
        print(f"Truncated prompt to {len(input_ids)} tokens")
    
    device = next(model.parameters()).device
    eos_id = tokenizer.token_to_id("<eos>")
    
    with torch.no_grad():
        for _ in range(min(max_length, config["seq_len"] - len(input_ids))):
            inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
            logits = model(inputs)[0, -1, :]
            
            # Apply penalties and sampling
            logits = apply_repetition_penalty(logits, inputs[0])
            logits = top_k_top_p_filtering(logits, config["top_k"], config["top_p"])
            logits = logits / max(temperature, 0.1)
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            if input_ids and next_id == input_ids[-1]:
                probs[next_id] = 0
                next_id = torch.multinomial(probs, num_samples=1).item()
            
            input_ids.append(next_id)
            if next_id == eos_id:
                break
    
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# =====================
# MAIN EXECUTION
# =====================
def main():
    tokenizer = initialize_tokenizer()
    model_path = os.path.join(config["forModel"], config["model_path"])
    
    # Training decision
    train_new = True
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        retr = input("Model exists. Retrain? (y/n): ").lower().strip()
        train_new = retr in ("y", "yes")
    
    # Training
    if train_new:
        use_ddp = config["use_ddp"] and torch.cuda.device_count() > 1
        world_size = torch.cuda.device_count() if use_ddp else 1
        
        if use_ddp:
            torch.multiprocessing.spawn(
                train_model,
                args=(tokenizer, world_size, config["forModel"], None),
                nprocs=world_size,
                join=True
            )
        else:
            train_model(0, tokenizer, world_size=1)
    
    # Load model
    model = NanoGPT(vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model.to(config["device"])
    
    # Compile model if possible
    if hasattr(torch, 'compile') and sys.platform != "win32":
        model = torch.compile(model)
    
    # Interactive generation
    print("\nGPT Ready! Type prompts below ('exit' to quit)")
    while True:
        try:
            prompt = input("\nPrompt: ")
            if prompt.lower().strip() in ('exit', 'quit', ""):
                break
            
            response = generate_text(model, tokenizer, prompt)
            print(f"Response: {response}")
            
            # Save to file
            if config["write_to_file"]:
                os.makedirs("output", exist_ok=True)
                file_count = len(glob.glob("output/*.txt"))
                with open(f"output/{file_count+1}.txt", "w", encoding="utf-8") as f:
                    f.write('\n'.join(textwrap.wrap(response, 80)))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
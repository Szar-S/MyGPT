import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders
import glob
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from config import config
from refresh import extract_text_from_pdfs_and_txts, create_tokenizer, token_save
from tqdm import tqdm 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

# =====================
# 1. TOKENIZER SETUP
# =====================
def initialize_tokenizer(forModel=None):
    if forModel is None:
        forModel = config["forModel"]
        
    tokenizer_path = os.path.join(forModel, config["bpe_tokenizer"])
    os.makedirs(forModel, exist_ok=True)
    data_path = os.path.join(forModel, config["data_corpus"])
    
    if not glob.glob(tokenizer_path) or os.path.getsize(tokenizer_path) == 0:
        if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:    
            text = extract_text_from_pdfs_and_txts()
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
        tokenizer = create_tokenizer(text)
    else:
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception as e:
            print(f"Error downlading tokenzier: {e}")
            input("Hit enter to create a new one.")
            if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:    
                text = extract_text_from_pdfs_and_txts()
            else:
                with open(data_path, "r", encoding="utf-8") as f:
                    text = f.read()
            tokenizer = create_tokenizer(text)
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer

# =====================
# 2. DATASET HANDLING
# =====================
class TextDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len or config["seq_len"]
        self.corpus_path = os.path.join(config["forModel"], config["data_corpus"])
        
        self.tokens = token_save(tokenizer, corpus_path)
        self.num_tokens = len(self.tokens)
        
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.pad_id = tokenizer.token_to_id("<pad>")

    def __len__(self):
        return max(1, self.num_tokens // self.seq_len)

    def __getitem__(self, idx):
        start_pos = idx * self.seq_len
        end_pos = start_pos + self.seq_len + 1

        if end_pos > self.num_tokens:
            pad_len = end_pos - self.num_tokens
            chunk = list(self.tokens[start_pos:]) + [self.pad_id] * pad_len
            # Set the last real token to <eos> if possible
            eos_pos = self.num_tokens - start_pos - 1
            if eos_pos >= 0 and eos_pos < len(chunk):
                chunk[eos_pos] = self.eos_id
        else:
            chunk = list(self.tokens[start_pos:end_pos])
        return {
            "input_ids": torch.tensor(chunk[:self.seq_len], dtype=torch.long),
            "labels": torch.tensor(chunk[1:self.seq_len+1], dtype=torch.long)
        }
# =====================
# 3. MODEL DEFINITION
# =====================
class NanoGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=None, n_layers=None, n_heads=None):
        super().__init__()
        if embed_size is None:
            embed_size = config["embed_size"]
        if n_layers is None:
            n_layers = config["n_layers"]
        if n_heads is None:
            n_heads = config["n_heads"]
        
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(config["seq_len"], embed_size)
        
        # Use TransformerEncoderLayer for GPT-style model
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=4*embed_size,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
    
    def forward(self, input_ids):
        device = input_ids.device
        _, seq_len = input_ids.shape
        
        # Create position IDs
        pos_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
        
        # Embed tokens and positions
        token_embeds = self.token_embed(input_ids)
        pos_embeds = self.pos_embed(pos_ids)
        x = token_embeds + pos_embeds
        
        # Apply causal mask for autoregressive behavior
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device=device)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, src_key_padding_mask=None, src_mask=mask)
        
        x = self.ln_final(x)
        return self.lm_head(x)

# =====================
# 4. TRAINING FUNCTION
# ====================
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Changed port to avoid conflicts
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_model(rank, tokenizer, world_size=0, forModel=config["forModel"], model=None):
    if config["use_ddp"] and world_size > 1:
        setup_ddp(rank, world_size)
        device = rank
        torch.cuda.set_device(device)
        print(torch.version.cuda)
    else:
        device = torch.device(config["device"])
    
    corpus_path = os.path.join(forModel, config["data_corpus"])
    save_path = os.path.join(forModel, config["model_path"])
    # Initialize dataset and dataloader
    if not os.path.exists(corpus_path) or os.path.getsize(corpus_path) == 0:
        extract_text_from_pdfs_and_txts()
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            pass
    
    dataset = TextDataset(corpus_path, tokenizer, config["seq_len"])
    if len(dataset) <= 300:
        print("ERROR: Not enough data to create even one training sample. "
              "Please check your PDF files or reduce seq_len.")
        input("Press Enter to exit...")
        return None
    

    # Initialize model
    if model is None:
        model = NanoGPT(
            vocab_size=tokenizer.get_vocab_size(),
            embed_size=config["embed_size"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"]
        )
    
    
    save_path = os.path.join(forModel, config["model_path"])
    
    # Load model if it exists
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        try:
            state_dict = torch.load(save_path, map_location=device)
            if rank == 0:
                print("Resuming training from existing model weights")
            model.load_state_dict(state_dict)
        except Exception as e:
            if rank == 0:
                print(f"Couldn't load existing weights: {e}. Training from scratch")
    else:
        if rank == 0:
            print("No existing model found. Training from scratch")
            
    model = model.to(device)
    
    if config["use_ddp"] and world_size > 1:
        model = DDP(model, device_ids=[rank])
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        scaler = torch.GradScaler(config["device"])
    except Exception as e: 
        print(f"error: {e}")
        scaler = None
    
    accumulation_steps = config["gradient_accumulation_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["learning_rate_decay"],
        eta_min=config["learning_rate"] / 10
    )
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    model.train()
    
    start_seq_len = config["start_seq_len"]
    seq_len_double_interval = config["seq_len_double_interval"]
    current_seq_len = start_seq_len
    
    original_seq_len = config["seq_len"]
    original_epochs = config["epochs"]
    
    global_epoch = 0
    total_epochs = original_epochs  # Start with config epochs
    
    # Epoch progress bar
    epoch_bar = tqdm(range(total_epochs), desc="Epochs", position=0)
    early_stop = False
    
    while global_epoch < total_epochs and not early_stop:
        if global_epoch > 0 and global_epoch % seq_len_double_interval == 0:
            current_seq_len = min(current_seq_len * 2, original_seq_len)
            if rank == 0:
                print(f"Epoch {global_epoch}: sequence length at {current_seq_len}")
        
        dataset.seq_len = current_seq_len
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank
        ) if config["use_ddp"] else None
        
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=config["num_workers"],
            persistent_workers=True
        )
        if sampler:
            sampler.set_epoch(global_epoch)
        total_loss = 0
        accumulation_counter = 0
        batch_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        desc=f"Epoch {global_epoch+1}/{total_epochs}",
                        position=1, leave=False
                        )
        
        optimizer.zero_grad()
        
        for batch_idx, batch in batch_bar:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    ) / accumulation_steps
            else:
                logits = model(inputs)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                ) / accumulation_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulation_counter += 1
            total_loss += loss.item() * accumulation_steps
            
            if accumulation_counter % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            batch_bar.set_postfix(
                loss=loss.item() * accumulation_steps, 
                avg_loss=total_loss/(batch_idx+1),
                seq_len=current_seq_len
            )
            
        scheduler.step()
        
        # Calculate average epoch loss
        epoch_avg_loss = total_loss / len(dataloader)
        
        # Check if we need to extend training
        if global_epoch == original_epochs - 1 and epoch_avg_loss > 1.0:
            # Loss still above 1.0 after initial epochs - add one more epoch
            if rank == 0:
                print(f"Loss {epoch_avg_loss:.4f} > 1.0 after {original_epochs} epochs - adding one more epoch")
            total_epochs += 1
            epoch_bar.total = total_epochs
            epoch_bar.refresh()
        
        if rank == 0:
            epoch_bar.update(1)
            epoch_bar.set_postfix(
                avg_loss=epoch_avg_loss,
                seq_len=current_seq_len,
                lr= scheduler.get_last_lr()[0]
            )
            print(f"Epoch {global_epoch+1} | Loss: {epoch_avg_loss:.4f} | Seq Len: {current_seq_len} | LR: {scheduler.get_last_lr()[0]:.7f}")
        
        global_epoch += 1
        
    if rank == 0:
        save_path = os.path.join(config["forModel"], config["model_path"])
        if config["use_ddp"] and world_size > 1:
            # DDP wrapped model
            state_dict = model.module.state_dict()
        else:
            # Regular model
            state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")
    
    if config["use_ddp"] and world_size > 1:
        dist.destroy_process_group()
    return model

# =====================
# 5. GENERATION FUNCTION
# =====================
def top_k_top_p_filtering(logits, top_k=config["top_k"], top_p=config["top_p"]):
    if top_k > 0:
        # Keep only top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    if top_p > 0.0:
        # Convert to probabilities
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter indices to original positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
    
    return logits

def generate_text(model, tokenizer, prompt, max_length=config["max_length"], temperature=config["temperature"]):
    """Generate text with top-k and top-p sampling from prompt"""
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_length):
        inputs = torch.tensor([input_ids], dtype=torch.long).to(next(model.parameters()).device)
        
        with torch.no_grad():
            logits = model(inputs)[0, -1, :]
            
        # Apply sampling techniques
        logits = top_k_top_p_filtering(
            logits,
            top_k=config["top_k"],
            top_p=config["top_p"]
        )
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        input_ids.append(next_id)
        if next_id == eos_id:
            break

    # Ensure proper spacing between tokens
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# =====================
# MAIN EXECUTION
# =====================
def main():
    tokenizer = initialize_tokenizer()
    model_path = os.path.join(config["forModel"], config["model_path"])
    model_exists = os.path.exists(model_path) and os.path.getsize(model_path) > 0
    use_ddp = config["use_ddp"] and torch.cuda.device_count() > 1
    
    # Determine if we need to train
    if model_exists:
        retr = input("Model exists. Retrain? (y/n): ").lower().strip()
        while retr not in ["y", "yes", "n", "no"]:
            retr = input("Please use yes(y) or no(n): ")
        retrain = (retr in ["y", "yes"])
    else:
        retrain = True
    
    # Training logic
    if retrain:
        if use_ddp:
            world_size = torch.cuda.device_count()
            torch.multiprocessing.spawn(
                train_model,
                args=(tokenizer, world_size, config["forModel"]),
                nprocs=world_size,
                join=True
            )
        else:
            train_model(0, tokenizer, world_size=1)
    
    # Load model for generation
    model = NanoGPT(vocab_size=tokenizer.get_vocab_size())
    
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model.to(config["device"])
    
    # Compile for faster inference
    if sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception as e:
            print("Could not compile model. Using uncompiled version:", e)
    else:
        print("Skipping torch.compile on Windows (no cl.exe).")
    
    # Interactive generation
    print("\nGPT Ready! Type your prompt (or 'quit' to exit)")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower().strip() in ["", "quit", "exit"]:
            break
        response = generate_text(model, tokenizer, prompt)
        print(f"Generated: {response}")

if __name__ == "__main__":
    main()

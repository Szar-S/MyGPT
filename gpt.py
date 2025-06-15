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
import textwrap

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
            print(f"Error loading tokenizer: {e}")
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
    def __init__(self, corpus_path, tokenizer, seq_len=None, split='train'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len or config["seq_len"]
        self.corpus_path = os.path.join(config["forModel"], config["data_corpus"])
        
        self.tokens = token_save(tokenizer, corpus_path)
        self.num_tokens = len(self.tokens)
        
        # Split dataset (90% train, 10% validation)
        split_idx = int(0.9 * self.num_tokens)
        if split == 'train':
            self.tokens = self.tokens[:split_idx]
        else:  # 'val'
            self.tokens = self.tokens[split_idx:]
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
        
        # Use dropout from config with default 0.0 for backward compatibility
        dropout_rate = config.get("dropout_rate", 0.0)
        
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(config["seq_len"], embed_size)
        
        # Transformer layers with residual connections
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=4*embed_size,
                dropout=dropout_rate,
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
# 4. TRAINING FUNCTION WITH OPTIMIZED SETTINGS
# ====================
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def evaluate(model, val_loader, loss_fn, device, rank):
    """Run validation on the model"""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    model.train()
    return avg_val_loss

def train_model(rank, tokenizer, world_size=0, forModel=config["forModel"], model=None):
    if config["use_ddp"] and world_size > 1:
        setup_ddp(rank, world_size)
        device = rank
        torch.cuda.set_device(device)
    else:
        device = torch.device(config["device"])
    
    corpus_path = os.path.join(forModel, config["data_corpus"])
    save_path = os.path.join(forModel, config["model_path"])
    
    # Initialize datasets
    if not os.path.exists(corpus_path) or os.path.getsize(corpus_path) == 0:
        extract_text_from_pdfs_and_txts()
    
    train_dataset = TextDataset(corpus_path, tokenizer, config["seq_len"], split='train')
    val_dataset = TextDataset(corpus_path, tokenizer, config["seq_len"], split='val')
    
    if len(train_dataset) <= 300:
        print("ERROR: Not enough data to create training samples.")
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
    
    # Load existing weights if available
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        try:
            state_dict = torch.load(save_path, map_location=device)
            if rank == 0:
                print("Resuming training from existing model weights")
            model.load_state_dict(state_dict)
        except:
            if rank == 0:
                print("Couldn't load existing weights. Training from scratch")
            
    model = model.to(device)
    
    if config["use_ddp"] and world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Add weight decay
    weight_decay = config.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6
    )
    
    accumulation_steps = config["gradient_accumulation_steps"]
    
    # Simpler but effective learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("lr_decay_steps", 1000),
        eta_min=config["learning_rate"] / 10
    )
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    model.train()
    
    # Curriculum learning setup
    start_seq_len = config["start_seq_len"]
    seq_len_double_interval = config["seq_len_double_interval"]
    current_seq_len = start_seq_len
    original_seq_len = config["seq_len"]
    
    # Training state tracking
    global_epoch = 0
    best_val_loss = float('inf')
    warmup_epochs = config.get("warmup_epochs", 3)
    
    # Dynamic epoch extension setup
    original_epochs = config["epochs"]
    total_epochs = original_epochs
    loss_below_one = False
    
    # Create validation dataloader
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"]
        )
    
    # Epoch progress bar
    epoch_bar = tqdm(total=total_epochs, desc="Epochs", position=0)
    
    while global_epoch < total_epochs:
        # Learning rate warmup
        if global_epoch < warmup_epochs:
            warmup_factor = min(1.0, (global_epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor
        
        # Update sequence length
        if global_epoch > 0 and global_epoch % seq_len_double_interval == 0:
            current_seq_len = min(current_seq_len * 2, original_seq_len)
            if rank == 0:
                print(f"Epoch {global_epoch}: sequence length at {current_seq_len}")
        
        # Update datasets with current sequence length
        train_dataset.seq_len = current_seq_len
        if rank == 0:
            val_dataset.seq_len = current_seq_len
        
        # Create distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        ) if config["use_ddp"] and world_size > 1 else None
        
        train_loader = DataLoader(
            train_dataset,
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
        batch_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {global_epoch+1}/{total_epochs}",
                        position=1, leave=False)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in batch_bar:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                ) / accumulation_steps
            
            loss.backward()
            accumulation_counter += 1
            total_loss += loss.item() * accumulation_steps
            
            # Gradient clipping to prevent explosions
            if accumulation_counter % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # Step after each gradient update
            
            batch_bar.set_postfix(
                loss=loss.item() * accumulation_steps, 
                avg_loss=total_loss/(batch_idx+1),
                seq_len=current_seq_len
            )
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Dynamic epoch extension logic
        if not loss_below_one:
            if avg_train_loss < 1.0:
                loss_below_one = True
                if rank == 0:
                    print(f"Loss dropped below 1.0 at epoch {global_epoch+1}")
            else:
                total_epochs += 1
                epoch_bar.total = total_epochs
                epoch_bar.refresh()
                if rank == 0:
                    print(f"Adding extra epoch (total now: {total_epochs})")
        
        # Validation
        val_loss = float('inf')
        if rank == 0:
            val_loss = evaluate(model, val_loader, loss_fn, device, rank)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, save_path)
                print(f"Saved best model (val_loss={val_loss:.4f})")
        
        # Update epoch progress
        if rank == 0:
            epoch_bar.update(1)
            epoch_bar.set_postfix(
                train_loss=avg_train_loss,
                val_loss=val_loss,
                seq_len=current_seq_len,
                lr=optimizer.param_groups[0]['lr'],
                extended_epochs=total_epochs - original_epochs
            )
            status = f"Epoch {global_epoch+1}/{total_epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}"
            if loss_below_one:
                status += " | loss < 1.0 achieved"
            else:
                status += f" | loss > 1.0 - Adding epochs"
            print(status)
        
        global_epoch += 1
    
    # Final model save
    if rank == 0:
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, save_path)
        print(f"Training complete. Final model saved to {save_path}")
    
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
    """Generate text with prompt truncation and sampling"""
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    
    # Truncate prompt if longer than model can handle
    max_ctx = config["seq_len"] - max_length - 1
    if len(input_ids) > max_ctx:
        input_ids = input_ids[-max_ctx:]
        print(f"Truncated prompt to {len(input_ids)} tokens")
    
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_length):
        inputs = torch.tensor([input_ids], dtype=torch.long).to(next(model.parameters()).device)
        
        with torch.no_grad():
            logits = model(inputs)[0, -1, :]
            
        # Apply sampling techniques
        logits = top_k_top_p_filtering(logits)
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
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
    model_exists = os.path.exists(model_path) and os.path.getsize(model_path) > 0
    use_ddp = config["use_ddp"] and torch.cuda.device_count() > 1
    
    # Determine if we need to train
    if model_exists:
        retr = input("Model exists. Retrain? (y/n): ").lower().strip()
        while retr not in ("y", "yes", "n", "no"):
            retr = input("Please use yes(y) or no(n): ")
        retrain = retr.startswith('y')
    else:
        retrain = True
    
    # Training logic
    if retrain:
        if use_ddp:
            world_size = torch.cuda.device_count()
            torch.multiprocessing.spawn(
                train_model,
                args=(tokenizer, world_size, config["forModel"], None),
                nprocs=world_size,
                join=True
            )
        else:
            train_model(0, tokenizer, world_size=1, model=None)
    
    # Load best model for generation
    model = NanoGPT(vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model.to(config["device"])
    
    # Compile for faster inference (if available)
    if sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception as e:
            print("Could not compile model. Using uncompiled version:", e)
    else:
        print("Skipping torch.compile on Windows")
    
    # Interactive generation
    print("\nGPT Ready! Type your prompt (or 'quit' to exit)")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower().strip() in ("", "quit", "exit"):
            break
        response = generate_text(model, tokenizer, prompt)
        print(f"Generated: {response}")
       
        os.makedirs("output", exist_ok=True)
        txt_path = os.path.join("output", "*.txt")
        txt_Files = glob.glob(txt_path)
        savedResponse = '\n'.join(textwrap.wrap(response, 80))
        if txt_Files:
            tempFiles = [os.path.basename(s).replace(".txt", "") for s in txt_Files]
            i = str(int(max(tempFiles)) + 1) + ".txt"
            i_path = os.path.join("output", i)
            os.makedirs(os.path.dirname(i_path), exist_ok=True)
            
            with open(i_path, "w", encoding="utf-8") as f:
                f.write(savedResponse)
        else:
            i = "1.txt"
            i_path = os.path.join("output", i)
            os.makedirs(os.path.dirname(i_path), exist_ok=True)
            
            with open(i_path, "w", encoding="utf-8") as f:
                f.write(savedResponse)

if __name__ == "__main__":
    main()

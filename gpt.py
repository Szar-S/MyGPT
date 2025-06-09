import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders
import glob
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from config import config
from refresh import extract_text_from_pdfs_and_txts, create_tokenizer
from tqdm import tqdm 
from torch.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

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
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    tokenizer.decoder = decoders.BPEDecoder()
    return tokenizer

# =====================
# 2. DATASET HANDLING
# =====================
class TextDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len or config["seq_len"]
        self.corpus_path = os.path.join(config["forModel"],config["data_corpus"]) 
        self.file_size = os.path.getsize(corpus_path)
        
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.pad_id = tokenizer.token_to_id("<pad>")
    def __len__(self):
        return max(1,self.file_size // (self.seq_len * 4))
    def __getitem__(self, idx):
        start_pos = idx * self.seq_len
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            f.seek(start_pos)
            text_chunk = f.read(self.seq_len * 10)
        tokens = [self.bos_id] + self.tokenizer.encode(text_chunk).ids
        if len(tokens) < self.seq_len + 1:
            tokens += [self.eos_id] + [self.pad_id] * (self.seq_len - len(tokens))
        
        input_ids = tokens[:self.seq_len]
        labels = tokens[1:self.seq_len + 1]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
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
        self.pos_embed = nn.Embedding(1024, embed_size)
        
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
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, src_key_padding_mask=None, src_mask=mask)
        
        x = self.ln_final(x)
        return self.lm_head(x)

# =====================
# 4. TRAINING FUNCTION
# =====================
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank,world_size=world_size)

def train_model(tokenizer, rank=0, world_size=1, forModel=config["forModel"]):
    if config["use_ddp"]:
        setup_ddp(rank, world_size)
        device = rank
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    corpus_path = os.path.join(forModel, config["data_corpus"])
    save_path = os.path.join(forModel, config["modal_path"])
    # Initialize dataset and dataloader
    if not os.path.exists(corpus_path) or os.path.getsize(corpus_path) == 0:
        text = extract_text_from_pdfs_and_txts()
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    dataset = TextDataset(corpus_path, tokenizer, config["seq_len"])
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size,rank=rank
    ) if config["use_ddp"] else None
    
    if len(dataset) <= 300:
        print("ERROR: Not enough data to create even one training sample. "
              "Please check your PDF files or reduce seq_len.")
        input("Press Enter to exit...")
        return None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers= 0
    )

    # Initialize model
    model = NanoGPT(
        vocab_size=tokenizer.get_vocab_size(),
        embed_size=config["embed_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"]
    )
    model = model.to(device)
    
    if config["use_ddp"]:
        model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # Epoch progress bar
    epoch_bar = tqdm(range(config["epochs"]), desc="Epochs", position=0)
    for epoch in epoch_bar:
        if sampler:
            sampler.set_epoch(epoch)
        total_loss = 0
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)
        for batch in batch_bar:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                )
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(batch_bar.n+1))
        if rank == 0:
            epoch_bar.set_postfix(avg_epoch_loss=total_loss/len(dataloader))
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
    
    if rank == 0:
        save_path = os.path.join(config["forModel"], config["modal_path"])
        torch.save(model.module.state_dict() if config["use_ddp"] else model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    if config["use_ddp"]:
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

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8):
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

    return tokenizer.decode(input_ids)

# =====================
# MAIN EXECUTION
# =====================
def main():
    # Initialize tokenizer (using your pre-trained BPE)
    tokenizer = initialize_tokenizer()
    model_path = os.path.join(config["forModel"], config["modal_path"])
    
    # Check if we should use distributed training
    use_ddp = config["use_ddp"] and torch.cuda.device_count() > 1
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print("Training model...")
        if use_ddp:
            # Start distributed training
            torch.multiprocessing.spawn(
                train_model,
                args=(tokenizer, torch.cuda.device_count()),
                nprocs=torch.cuda.device_count(),
                join=True
            )
            model = NanoGPT(vocab_size=tokenizer.get_vocab_size())
            model.load_state_dict(torch.load(model_path))
        else:
            # Single process training
            model = train_model(tokenizer)
    else:
        print("Loading pre-trained model...")
        model = NanoGPT(vocab_size=tokenizer.get_vocab_size())
        model.load_state_dict(torch.load(model_path))
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Interactive generation
    print("\nGPT Ready! Type your prompt (or 'quit' to exit)")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        
        response = generate_text(model, tokenizer, prompt)
        print(f"Generated: {response}")

if __name__ == "__main__":
    main()
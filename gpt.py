import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers, models
import glob
from torch.utils.data import Dataset, DataLoader
from config import config
from refresh import extract_text_from_pdfs_and_txts, create_tokenizer
from tqdm import tqdm 
from torch.cuda.amp import GradScaler

# =====================
# 1. TOKENIZER SETUP
# =====================
def initialize_tokenizer(forModel=None):
    if forModel is None:
        forModel = config["forModel"]
    tokenizer_path = os.path.join(forModel, config["bpe_tokenizer"])
    if not glob.glob(tokenizer_path) or os.path.getsize(tokenizer_path) == 0:
        data_path = os.path.join(forModel, config["data_corpus"])
        if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
            os.makedirs(forModel, exist_ok=True)
            text = extract_text_from_pdfs_and_txts()
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
        create_tokenizer(text)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = decoders.BPEDecoder()
    return tokenizer

# =====================
# 2. DATASET HANDLING
# =====================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=None):
        self.tokenizer = tokenizer
        if seq_len is None:
            seq_len = config["seq_len"]
        # Encode the entire text as token IDs
        self.tokens = tokenizer.encode(text).ids
        # Add special tokens
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        self.tokens = [bos_id] + self.tokens + [eos_id]
        # Split into sequences of length seq_len
        self.seq_len = seq_len
        self.samples = []
        for i in range(0, len(self.tokens) - seq_len, seq_len):
            input_ids = self.tokens[i:i+seq_len]
            labels = self.tokens[i+1:i+seq_len+1]
            if len(labels) < seq_len:
                continue
            self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, labels = self.samples[idx]
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
        batch_size, seq_len = input_ids.shape
        
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
def train_model(tokenizer, forModel=None):
    if forModel is None:
        forModel = config["forModel"]
    corpus_path = os.path.join(forModel, config["data_corpus"])
    save_path = os.path.join(forModel, config["modal_path"])
    # Initialize dataset and dataloader
    if not os.path.exists(corpus_path) or os.path.getsize(corpus_path) == 0:
        text = extract_text_from_pdfs_and_txts()
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
    dataset = TextDataset(text, tokenizer, config["seq_len"])
    if len(dataset) <= 300:
        print("ERROR: Not enough data to create even one training sample. "
              "Please check your PDF files or reduce seq_len.")
        input("Press Enter to exit...")
        return None
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # Initialize model
    model = NanoGPT(
        vocab_size=tokenizer.get_vocab_size(),
        embed_size=config["embed_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler = GradScaler()
    import time
    sample_batch = next(iter(dataloader))
    inputs = sample_batch["input_ids"].to(device)
    targets = sample_batch["labels"].to(device)
    model.train()
    start_time = time.time()
    optimizer.zero_grad()
    if use_amp:
        with torch.amp.autocast(device_type='cuda'):
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
    end_time = time.time()
    batch_time = end_time - start_time
    num_batches = len(dataloader)
    epochs = config["epochs"]
    estimated_total = batch_time * num_batches * epochs
    mins, secs = divmod(int(estimated_total), 60)
    print(f"Estimated training time for {epochs} epochs: {mins} min {secs} sec")

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch", leave=True, dynamic_ncols=True)
        for batch in batch_iter:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(inputs)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            batch_iter.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(dataloader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

# =====================
# 5. GENERATION FUNCTION
# =====================
def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8):
    """Generate text from prompt"""
    model.eval()
    input_ids = tokenizer.encode(prompt).ids

    eos_token = "<eos>"
    eos_token_id = tokenizer.token_to_id(eos_token)

    for _ in range(max_length):
        with torch.no_grad():
            inputs = torch.tensor([input_ids], dtype=torch.long)
            logits = model(inputs)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        input_ids.append(next_id)
        if next_id == eos_token_id:  # Use eos_token_id for stopping
            break

    return tokenizer.decode(input_ids)

# =====================
# MAIN EXECUTION
# =====================
def main():
    # Initialize tokenizer (using your pre-trained BPE)
    tokenizer = initialize_tokenizer()
    model_path = os.path.join(config["forModel"], config["modal_path"])
    
    # Check if model exists
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print("Training model...")
        model = train_model(tokenizer)
        if model is None:
            print("Model training failed. Exiting.")
            return
    else:
        print("Loading pre-trained model...")
        model = NanoGPT(
            vocab_size=tokenizer.get_vocab_size(),
            embed_size=config["embed_size"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"]
        )
        model.load_state_dict(torch.load(model_path))
    
    # Interactive generation loop
    print("\nGPT Ready! Type your prompt (or 'quit' to exit)")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        
        # Generate and print response
        response = generate_text(model, tokenizer, prompt)
        print(f"Generated: {response}")

if __name__ == "__main__":
    main()
import torch
config = {
    "forModel": "forModel",
    "forData": "data",
    "model_path": "gpt_model.pth",
    "data_corpus": "data_corpus.txt",
    "bpe_tokenizer": "bpe_tokenizer.json",
    "min_data_size": 300,        # int
    "max_length": 200,            # int
    
    # Model Architecture 
    "vocab_size": 10000,         # int
    "embed_size": 216,           # int
    "n_layers": 6,              # int
    "n_heads": 6,                # int
    
    # Training Parameters
    "seq_len": 256,              # int
    "start_seq_len": 32,
    "batch_size": 8,             # int
    "epochs": 15,                 # int
    "seq_len_double_interval":5,
    "learning_rate": 2e-4,       # float
    "gradient_accumulation_steps": 4,
    "learning_rate_decay": 0.95,
    
    # Generation Settings
    "top_k": 40,                 # int
    "top_p": 0.85,                # float
    "temperature": 0.7,           # float
    
    # System
    "use_ddp": False,            # bool
    "num_workers": 4,            #int
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

import torch
config = {
    "forModel": "forModel",
    "forData": "data",
    "model_path": "gpt_model.pth",
    "data_corpus": "data_corpus.txt",
    "bpe_tokenizer": "bpe_tokenizer.json",
    
    # Model Architecture 
    "vocab_size": 20000,         # int
    "embed_size": 128,           # int
    "n_layers": 4,               # int
    "n_heads": 4,                # int
    "drapout_rate": 0.1,                
    
    # Training Parameters
    "seq_len": 256,              # int
    "start_seq_len": 32,
    "batch_size": 16,             # int
    "epochs": 10,                 # int
    "seq_len_double_interval":4,
    "learning_rate": 3e-4,       # float
    "gradient_accumulation_steps": 2,
    "learning_rate_decay": 0.9,
    "weight_decay": 0.01,
    "warmup_epochs":2,
    "patience": 3,
    
    # Generation Settings
    "top_k": 40,                 # int
    "top_p": 0.85,                # float
    "temperature": 0.7,           # float
    "min_data_size": 300,        # int
    "max_length": 250,            # int
    
    # System
    "use_ddp": True,            # bool
    "num_workers": 4,            #int
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "world_size": 3
}

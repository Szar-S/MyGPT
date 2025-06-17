import torch
config = {
    "forModel": "forModel",
    "forData": "data",
    "model_path": "gpt_model.pth",
    "data_corpus": "data_corpus.txt",
    "bpe_tokenizer": "bpe_tokenizer.json",
    
    # Model Architecture 
    "vocab_size": 20000,         # int
    "embed_size": 132,           # int
    "n_layers": 3,               # int
    "n_heads": 6,                # int
    "drapout_rate": 0.4,                
    
    # Training Parameters
    "seq_len": 512,              # int
    "start_seq_len": 64,
    "batch_size": 3,             # int
    "epochs":16,                 # int
    "seq_len_double_interval":4,
    "learning_rate": 5e-4,       # float
    "gradient_accumulation_steps": 4,
    "lr_decay_steps": 500,
    "learning_rate_decay": 0.9,
    "weight_decay": 0.01,
    "warmup_epochs":4,
    
    # Generation Settings
    "top_k": 50,                 # int
    "top_p": 0.92,                # float
    "temperature": 0.7,           # float
    "min_data_size": 300,        # int
    "max_length": 150,            # int
    "include_prompt": False,
    "repeat_generate": True,
    "repeat_int": 10,
    "write_to_file": True,
    
    # System
    "use_ddp": True,            # bool
    "num_workers": 4,            #int
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "world_size": 3
}

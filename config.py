config = {
    "forModel": "forModel",
    "forData": "data",
    "model_path": "gpt_model.pth",
    "data_corpus": "data_corpus.txt",
    "bpe_tokenizer": "bpe_tokenizer.json",
    "min_data_size": 300,        # int
    "max_length": 50,            # int
    
    # Model Architecture 
    "vocab_size": 10000,         # int
    "embed_size": 192,           # int
    "n_layers": 6,              # int
    "n_heads": 6,                # int
    
    # Training Parameters
    "seq_len": 128,              # int
    "batch_size": 64,             # int
    "epochs": 15,                 # int
    "learning_rate": 3e-4,       # float
    
    # Generation Settings
    "top_k": 40,                 # int
    "top_p": 0.85,                # float
    "temperature": 0.7,           # float
    
    # System
    "use_ddp": False,            # bool
    "num_workers": 4             #int
}

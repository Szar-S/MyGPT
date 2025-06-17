import torch
config = {
    "forModel": "forModel",
    "forData": "data",
    "model_path": "gpt_model.pth",
    "data_corpus": "data_corpus.txt",
    "bpe_tokenizer": "bpe_tokenizer.json",
    
    # Model Architecture
    "vocab_size": 8000,
    "embed_size": 128,
    "n_layers": 4,
    "n_heads": 4,
    "dropout_rate": 0.3,
    
    # Training Parameters
    "seq_len": 256,
    "start_seq_len": 64,
    "batch_size": 4,
    "epochs": 24,
    "seq_len_double_interval": 6,
    "learning_rate": 3e-4,
    "gradient_accumulation_steps": 8,
    "lr_decay_steps": 1000,
    "weight_decay": 0.05,
    "warmup_epochs": 6,
    
    # Generation Settings
    "top_k": 30,
    "top_p": 0.85,
    "temperature": 0.8,
    "min_data_size": 100,
    "max_length": 200,
    "include_prompt": False,
    "repeat_generate": True,
    "repeat_int": 5,
    "write_to_file": True,
    
    # System Features
    "use_ddp": False,
    "use_amp": True,
    "num_workers": 2,
    "repetition_penalty": 1.25,
    "checkpoint_every": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
config = {
    "forModel": "forModel",
    "forData": "data",
    "modal_path": "gpt_model.pth",
    "data_corpus": "data_corpus.txt",
    "bpe_tokenizer": "bpe_tokenizer.json",
    
    "vocab_size": 50257,         # int
    "seq_len": 32,              # int
    "batch_size": 32,             # int
    "epochs": 8,                 # int
    "learning_rate": 6e-4,       # float
    "embed_size": 256,           # int
    "n_layers": 32,              # int
    "n_heads": 8,                # int
    "use_ddp": False,            # bool
    "top_k": 50,                 # int
    "top_p": 0.9,                # float
    "temperature": 0.8           # float
}

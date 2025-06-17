# MyGPT

A minimal, self-hosted GPT-style text generation model for local experimentation.

## Features

- **Curriculum Learning** - Sequence length starts at 64 tokens and doubles during training for faster convergence
- Extracts text from PDFs and TXT files in the `data/` directory
- Trains a Byte-Pair Encoding (BPE) tokenizer using `tokenizers` library (with custom pre-tokenization)
- Implements a GPT-like transformer model in PyTorch with pre-normalization
- Interactive text generation with repetition penalty
- Supports top-k and top-p (nucleus) sampling for generation
- Distributed training support via DDP (multi-GPU)
- Visual training progress with per-epoch and per-batch loss
- Memory-mapped token storage for efficient dataset loading
- Generated text saved in `output/` directory
- Learning rate warmup and decay
- Gradient accumulation support
- Torch.compile optimization for faster inference (Linux only)
- Special tokens: `<unk>`, `<pad>`, `<bos>`, `<eos>`, `<sep>`, `<doc>`

## Setup

1. **Add Data**  
   Place your `.pdf` and `.txt` files in the `data/` directory.

2. **Install Requirements**  
   Install required Python packages:
```sh
pip install torch tokenizers pdfplumber tqdm numpy
```

3. **First Run / Training**
   - Run main script to train or load model:
```sh
python gpt.py
```
   - Training starts with 64-token sequences and doubles every 4 epochs
   - Training progress shows current sequence length and loss metrics

4. **Interactive Generation**
   - Enter prompts and receive generated text using top-k/top-p sampling
   - Generated responses saved in `output/` as numbered text files

## Distributed Training
- Enable with `"use_ddp": True` in `config.py`
- Automatically uses available GPUs
- Supports multi-node training via NCCL backend
- Default world size: 3 GPUs

## Updating Data
To update data files:
```sh
python refresh.py  # Rebuilds corpus and tokenizer
python gpt.py      # Retrain model
```

## File Structure

- `gpt.py` — Main script: model, training, generation
- `refresh.py` — Extracts text from PDFs/TXT and trains tokenizer
- `config.py` — Configuration for paths and hyperparameters
- `forModel/` — Stores model, tokenizer, memory-mapped tokens
- `data/` — Source `.pdf` and `.txt` files
- `output/` — Generated text outputs
- `test_gpt.py` — Unit tests for model components

## Configuration
All parameters set in `config.py`:

**Curriculum Learning**
- `start_seq_len`: Initial sequence length (64)
- `seq_len_double_interval`: Epochs before doubling length (4)
- `warmup_epochs`: Learning rate warmup period (4)

**Model Architecture**
- `vocab_size`: Vocabulary size (20000)
- `embed_size`: Embedding dimension (132)
- `n_layers`: Transformer layers (3)
- `n_heads`: Attention heads (6)
- `drapout_rate`: Regularization rate (0.4)

**Training Parameters**
- `seq_len`: Maximum sequence length (512)
- `batch_size`: Training batch size (3)
- `epochs`: Training epochs (16)
- `learning_rate`: Optimizer rate (5e-4)
- `gradient_accumulation_steps`: Accumulation steps (4)
- `weight_decay`: L2 regularization (0.01)

**Generation Settings**
- `top_k`: Top-k sampling (50)
- `top_p`: Nucleus sampling (0.92)
- `temperature`: Sampling temperature (0.7)
- `max_length`: Max generation length (150)
- `repeat_generate`: Apply repetition penalty (True)
- `repeat_int`: Penalty interval (10 tokens)

**System**
- `use_ddp`: Distributed training (True)
- `num_workers`: Data loader workers (4)
- `device`: Auto-detects GPU
- `world_size`: DDP processes (3)

## Notes

* Curriculum learning starts with 64-token sequences
* Memory-mapped token files (`.tokens`) enable fast dataset loading
* Tokenizer uses custom pre-tokenization (whitespace/punctuation/digits)
* Training includes 90/10 train/validation split
* Generation applies repetition penalty to reduce looping
* Model automatically compiles with `torch.compile` on Linux
* PDF extraction uses pdfplumber for better text handling
* Text cleaning handles OCR artifacts and special characters
* Includes unit tests for core functionality (`test_gpt.py`)

## License

MIT License © Szar-S
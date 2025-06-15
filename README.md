# MyGPT

A minimal, self-hosted GPT-style text generation model for local experimentation.

## Features

- **Curriculum Learning** - Sequence length starts small (32 tokens) and doubles during training for faster convergence
- Extracts text from PDFs and TXT files in the `data/` directory
- Trains a Byte-Pair Encoding (BPE) tokenizer using the `tokenizers` library (with ByteLevel for proper spacing)
- Implements a simple GPT-like transformer model in PyTorch
- Interactive text generation from the command line
- Supports top-k and top-p (nucleus) sampling for generation
- Distributed training support via DDP (multi-GPU)
- Visual training progress with per-epoch and per-batch loss
- Ignores padding tokens in loss for stable training
- Easy retraining and data refresh
- Memory-mapped token storage for efficient dataset loading
- Dynamic epoch extension (automatically extends training if loss > 1.0)
- Automatic prompt truncation during generation
- Generated text saved in `output/` directory
- Learning rate warmup and decay
- Gradient accumulation support
- Torch.compile optimization for faster inference (Linux only)

## Setup

1. **Add Data**  
   Place your `.pdf` and `.txt` files in the `data/` directory.

2. **Install Requirements**  
   Install the required Python packages:
```sh
pip install torch tokenizers PyPDF2 tqdm numpy
```

3. **First Run / Training**
   - Run the main script to train or load the model:
```sh
python gpt.py
```
   - If no model exists, you will be prompted to train a new one
   - Training uses curriculum learning (sequence length starts at 32 and doubles during training)
   - If a model exists, it will be loaded for text generation
   - Training progress shows current sequence length and loss metrics

4. **Interactive Generation**
   - After training or loading, enter prompts and receive generated text using top-k/top-p sampling
   - Generated responses are automatically saved in the `output/` directory as numbered text files

## Distributed Training (Optional)
- If you have multiple GPUs and want to use Distributed Data Parallel (DDP):
  - Set `"use_ddp": True` in `config.py`
  - The script will automatically use all available GPUs for training
  - Supports multi-node training via NCCL backend

## Updating Data
If you add new data files or remove old ones:

1. Run the refresh script to rebuild the corpus and tokenizer:
```sh
python refresh.py
```

2. (Optional but recommended) Backup your previous model file (`forModel/gpt_model.pth`)

3. Retrain the model with the updated data:
```sh
python gpt.py
```

## File Structure

- `gpt.py` — Main script: model, training, and generation loop
- `refresh.py` — Extracts text from PDFs/TXT and (re)trains tokenizer
- `config.py` — Configuration dictionary for paths and hyperparameters
- `forModel/` — Stores trained model, tokenizer, and memory-mapped tokens
- `data/` — Place your source `.pdf` and `.txt` files here
- `output/` — Directory for saving generated text outputs
- `test_gpt.py` — Unit tests for model components

## Configuration

All paths and hyperparameters are set in `config.py`.

**Curriculum Learning**
- `start_seq_len`: Initial sequence length (default: 32)
- `seq_len_double_interval`: Epochs before doubling sequence length (default: 6)
- `warmup_epochs`: Learning rate warmup period (default: 4)

**Model Architecture**
- `vocab_size`: Vocabulary size (default: 10000)
- `embed_size`: Embedding dimension (default: 192)
- `n_layers`: Number of transformer layers (default: 6)
- `n_heads`: Number of attention heads (default: 6)
- `dropout_rate`: Regularization rate (default: 0.1)

**Training Parameters**
- `seq_len`: Maximum sequence length (default: 256)
- `batch_size`: Training batch size (default: 8)
- `epochs`: Minimum training epochs (default: 3)
- `learning_rate`: Optimizer learning rate (default: 1e-4)
- `gradient_accumulation_steps`: Steps for gradient accumulation (default: 4)
- `lr_decay_steps`: Steps before learning rate decay (default: 500)
- `weight_decay`: L2 regularization strength (default: 0.01)

**Generation Settings**
- `top_k`: Top-k sampling parameter (default: 40)
- `top_p`: Nucleus sampling parameter (default: 0.85)
- `temperature`: Sampling temperature (default: 0.7)
- `max_length`: Maximum generation length (default: 250)

**System**
- `use_ddp`: Enable distributed training (default: True)
- `num_workers`: Data loader workers (default: 4)
- `device`: Training device (auto-detects GPU)
- `world_size`: DDP world size (default: 3)

## Notes

* Training uses curriculum learning for faster convergence - sequence length starts small and increases during training

* Memory-mapped token files (`.tokens`) are generated for faster dataset loading

* Padding tokens are ignored in loss calculation for more stable training

* The model and tokenizer are always kept in sync; if you refresh data or tokenizer, retrain the model

* Training and generation work on both CPU and GPU (DDP supported for multi-GPU setups)

* Tokenizer uses ByteLevel pre-tokenizer and decoder for correct spacing in generated text

* Last real token in sequences is automatically set to `<eos>` during training

* Includes unit tests for core functionality (`test_gpt.py`)

* Training automatically extends if loss remains above 1.0

* Generated text is saved in `output/` directory as numbered text files

* Torch.compile optimization is automatically applied on Linux systems

## License

MIT License © Szar-S
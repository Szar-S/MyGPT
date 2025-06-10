# MyGPT

A minimal, self-hosted GPT-style text generation model for local experimentation.

## Features

- **Curriculum Learning** - Sequence length starts small (32 tokens) and doubles during training for faster convergence
- Extracts text from PDFs and TXT files in the `data/` directory
- Trains a Byte-Pair Encoding (BPE) tokenizer using the `tokenizers` library (with ByteLevel for proper spacing)
- Implements a simple GPT-like transformer model in PyTorch
- Interactive text generation from the command line
- Supports top-k and top-p (nucleus) sampling for generation
- Distributed training (optional, via DDP)
- Visual training progress with per-epoch and per-batch loss
- Ignores padding tokens in loss for stable training
- Easy retraining and data refresh
- Memory-mapped token storage for efficient dataset loading

## Setup

1. **Add Data**  
   Place your `.pdf` and `.txt` files in the `data/` directory.

2. **Install Requirements**  
 * Install the required Python packages:
```sh
pip install torch tokenizers PyPDF2 tqdm numpy
```
3. **First Run / Training**
* Run the main script to train or load the model:
```sh
python gpt.py
```
* If no model exists, you will be prompted to train a new one
* Training uses curriculum learning (sequence length starts at 32 and doubles during training)
* If a model exists, it will be loaded for text generation
* Training progress shows current sequence length and loss metrics
4. **Interactive Generation**
* After training or loading, enter prompts and receive generated text using top-k/top-p sampling.

## Distributed Training (Optional)
* If you have multiple GPUs and want to use Distributed Data Parallel (DDP):
* Set `"use_ddp": True` in `config.py`
* The script will automatically use all available GPUs for training
* Supports multi-node training via NCCL backend

## Updating Data
* If you add new data files or remove old ones:

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
- `forModel/` —  Stores trained model, tokenizer, and memory-mapped tokens
- `data/` — Place your source `.pdf` and `.txt` files here
- `test_gpt.py` — Unit tests for model components

## Configuration

All paths and hyperparameters are set in `config.py`.
**Curriculum Learning**
- `start_seq_len`: Initial sequence length (default: 32)
- `seq_len_double_interval`: Epochs before doubling sequence length (default: 5)

**Model Architecture**
- `embed_size`: Embedding dimension (default: 216)
- `n_layers`: Number of transformer layers (default: 6)
- `n_heads`: Number of attention heads (default: 6)

**Training Parameters**
- `seq_len`: Maximum sequence length (default: 256)
- `batch_size`: Training batch size (default: 8)
- `epochs`: Total training epochs (default: 15)
- `learning_rate`: Optimizer learning rate (default: 2e-4)

**Generation Settings**
- `top_k`: Top-k sampling parameter (default: 40)
- `top_p`: Nucleus sampling parameter (default: 0.85)
- `temperature`: Sampling temperature (default: 0.7)

## Notes
* Training uses curriculum learning for faster convergence - sequence length starts small and increases during training

* Memory-mapped token files (`.tokens`) are generated for faster dataset loading

* Padding tokens are ignored in loss calculation for more stable training

* The model and tokenizer are always kept in sync; if you refresh data or tokenizer, retrain the model

* Training and generation work on both CPU and GPU (DDP supported for multi-GPU setups)

* Tokenizer uses ByteLevel pre-tokenizer and decoder for correct spacing in generated text

* Last real token in sequences is automatically set to `<eos>` during training

* Includes unit tests for core functionality (`test_gpt.py`)

## License

MIT License © Szar-S
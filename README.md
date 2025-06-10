# MyGPT

A minimal, self-hosted GPT-style text generation model for local experimentation.

## Features

- Extracts text from PDFs and TXT files in the `data/` directory
- Trains a Byte-Pair Encoding (BPE) tokenizer using the `tokenizers` library
- Implements a simple GPT-like transformer model in PyTorch
- Interactive text generation from the command line
- Supports top-k and top-p (nucleus) sampling for generation
- Distributed training (optional, via DDP)
- Visual training progress with per-epoch and per-batch loss
- Easy retraining and data refresh

## Setup

1. **Add Data**  
   Place your `.pdf` and `.txt` files in the `data/` directory.

2. **Install Requirements**  
   Install the required Python packages:
   ```sh
   pip install torch tokenizers PyPDF2 tqdm numpy
   ```

3. **First Run / Training**  
   Run the main script to train or load the model:
   ```sh
   python gpt.py
   ```
   - If no model exists, you will be prompted to train a new one.
   - If a model exists, it will be loaded for text generation.
   - Training progress is shown with visual indicators (progress bars for epochs and batches, live loss reporting).

4. **Interactive Generation**  
   After training or loading, you can enter prompts and receive generated text using top-k/top-p sampling.

## Distributed Training (Optional)

If you have multiple GPUs and want to use Distributed Data Parallel (DDP):

- Set `"use_ddp": True` in `config.py`.
- The script will automatically use all available GPUs for training.

## Updating Data

If you add new data files or remove old ones:

1. Run the refresh script to rebuild the corpus and tokenizer:
   ```sh
   python refresh.py
   ```

2. (Optional but recommended) Backup your previous model file (`forModel/gpt_model.pth`).

3. Retrain the model with the updated data:
   ```sh
   python gpt.py
   ```

## File Structure

- `gpt.py` — Main script: model, training, and generation loop
- `refresh.py` — Extracts text from PDFs/TXT and (re)trains the tokenizer
- `config.py` — Configuration dictionary for paths and hyperparameters
- `forModel/` — Stores the trained model and tokenizer
- `data/` — Place your source `.pdf` and `.txt` files here

## Configuration

All paths and hyperparameters are set in `config.py`.  
You can adjust:
- Model size (`embed_size`, `n_layers`, `n_heads`)
- Training parameters (`batch_size`, `epochs`, `learning_rate`)
- Tokenizer settings (`vocab_size`)
- Generation settings (`top_k`, `top_p`, `temperature`)
- DDP usage (`use_ddp`)
- DataLoader workers (`num_workers`)

## Notes

- Padding tokens are ignored in the loss calculation for more stable training.
- The model and tokenizer are always kept in sync; if you refresh data or tokenizer, retrain the model.
- Training and generation work on both CPU and GPU. DDP is supported for multi-GPU setups.

## License

MIT License © Szar-S
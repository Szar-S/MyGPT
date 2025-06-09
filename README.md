# MyGPT

A minimal, self-hosted GPT-style text generation model for local experimentation.

## Features

- Extracts text from PDFs and TXT files in the `data/` directory
- Trains a Byte-Pair Encoding (BPE) tokenizer
- Implements a simple GPT-like transformer model in PyTorch
- Interactive text generation from the command line

## Usage

1. Place your `.pdf` and `.txt` files in the `data/` directory.
2. Install requirements:
    ```sh
    pip install torch tokenizers PyPDF2 tqdm
    ```
3. Run the main script:
    ```sh
    python gpt.py
    ```
4. Follow the prompts to train or load the model and generate text.

5. If you added new data or deleted previous data run the updater script:
    ```sh
    python refresh.py
    ```
6. Then backup the previous model stored in forModel named gpt_model. Run the main script to train:
    ```sh
    python gpt.py
    ```

## License

MIT License © Szar-S

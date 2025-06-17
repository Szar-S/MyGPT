import os
import glob
import re
import logging
import numpy as np
from config import config
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text):
    # Remove non-printable characters except common punctuation
    text = re.sub(r'[^\x20-\x7E\u00A0-\u00FF\u2018-\u201F]', ' ', text)
    
    # Normalize whitespace and special characters
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'[‐‑‒–—―]', '-', text)  # Normalize hyphens/dashes
    text = re.sub(r'[“”]', '"', text)  # Normalize quotes
    text = re.sub(r'[‘’]', "'", text)   # Normalize apostrophes
        
    # Fix common OCR artifacts
    text = re.sub(r'\b(\w+)\s+-\s+(\w+)\b', r'\1-\2', text)  # Fix hyphenated words
    text = re.sub(r'\b(\w+)\s*=\s*(\w+)\b', r'\1=\2', text)  # Fix equals signs
    
    # Remove orphaned characters
    text = re.sub(r'\s[^\w\s]\s', ' ', text)
    
    # Handle bullet points and numbered lists
    text = re.sub(r'\s*[\u2022\u25E6]\s*', '\n• ', text)
    text = re.sub(r'\s*\d+\.\s+', '\n1. ', text)
    
    text = re.sub(r'(?<!\w)\.(?!\w)', ' ', text)
    text = re.sub(r'\s([.,!?;:](?:\s|$))', r'\1', text)
    
    # Final cleanup
    text = text.strip()
    return text

def extract_text_from_pdfs_and_txts(folder=None, forModel=None):
    if folder is None:
        folder = config["forData"]
    if forModel is None:
        forModel = config["forModel"]
    
    os.makedirs(folder, exist_ok=True)
    os.makedirs(forModel, exist_ok=True)
    
    textAll = []
    pdf_path = os.path.join(folder, "*.pdf")
    txt_path = os.path.join(folder, "*.txt")
    data_corpus_path = os.path.join(forModel, config["data_corpus"])
    
    pdf_files = glob.glob(pdf_path)
    txt_files = glob.glob(txt_path)
    
    total_files = len(pdf_files) + len(txt_files)
    processed_files = 0
    
    # PDF Extraction with improved handling
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF files in '{folder}' directory")
        try:
            import pdfplumber  # More reliable text extraction
            logger.info("Using pdfplumber for PDF extraction")
        except ImportError:
            import PyPDF2
            logger.warning("pdfplumber not installed, falling back to PyPDF2")
            pdfplumber = None
        
        for filename in pdf_files:
            try:
                logger.info(f"Processing PDF: {os.path.basename(filename)}")
                with open(filename, "rb") as f:
                    if pdfplumber:
                        with pdfplumber.open(f) as pdf:
                            for i, page in enumerate(pdf.pages):
                                page_text = page.extract_text()
                                if page_text:
                                    textAll.append(page_text)
                                if (i+1) % 50 == 0:
                                    logger.info(f"Processed {i+1} pages")
                    else:
                        reader = PyPDF2.PdfReader(f)
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            if page_text:
                                textAll.append(page_text)
                            if (i+1) % 50 == 0:
                                logger.info(f"Processed {i+1} pages")
                processed_files += 1
                logger.info(f"Completed: {os.path.basename(filename)}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    else:
        logger.info(f"No PDF files found in '{folder}' directory")
    
    # Text File Extraction with encoding fallback
    if txt_files:
        logger.info(f"Found {len(txt_files)} TXT files in '{folder}' directory")
        for filename in txt_files:
            try:
                logger.info(f"Processing TXT: {os.path.basename(filename)}")
                # Try multiple encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(filename, "r", encoding=encoding) as f:
                            text = f.read()
                            textAll.append(text)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.warning(f"Could not decode {filename} with common encodings")
                processed_files += 1
                logger.info(f"Completed: {os.path.basename(filename)}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    else:
        logger.info(f"No TXT files found in '{folder}' directory")
    
    # Early exit if no files processed
    if not textAll:
        logger.error("No text extracted from any files")
        return ""
    
    logger.info(f"Processed {processed_files}/{total_files} files successfully")
    
    # Combine all text with separator
    combined_text = "\n\n".join(textAll)

    cleaned_text = clean_text(combined_text)
    
    # Save cleaned corpus
    if cleaned_text.strip():
        logger.info(f"Cleaned text size: {len(cleaned_text):,} characters")
        
        os.makedirs(os.path.dirname(data_corpus_path), exist_ok=True)
        with open(data_corpus_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        
        logger.info(f"Data corpus saved to: {data_corpus_path}")
        return cleaned_text
    else:
        logger.error("No valid text after cleaning")
        return ""

def create_tokenizer(text, forModel=None):
    if forModel is None:
        forModel = config["forModel"]
    os.makedirs(forModel, exist_ok=True)
        
    tokenizer_path = os.path.join(forModel, config["bpe_tokenizer"])
    
    # Special tokens with descriptions
    special_tokens = [
        "<unk>",  # Unknown tokens
        "<pad>",  # Padding
        "<bos>",  # Beginning of sequence
        "<eos>",  # End of sequence
        "<sep>",  # Separation token
        "<doc>"   # Document separator
    ]
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train or load tokenizer
    if not os.path.exists(tokenizer_path) or os.path.getsize(tokenizer_path) == 0:
        logger.info("Training new tokenizer")
        
        # Train on chunks to handle large texts
        chunk_size = 10000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        tokenizer.train_from_iterator(chunks, trainer=trainer)
        
        tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
    else:
        logger.info("Loading existing tokenizer")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Configure decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Add missing special tokens
    for token in special_tokens:
        if tokenizer.token_to_id(token) is None:
            tokenizer.add_special_tokens([token])
    
    tokenizer.save(tokenizer_path)
    
    # Log tokenizer info
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Tokenizer ready. Vocabulary size: {vocab_size}")
    
    return tokenizer

def token_save(tokenizer, corpus_path):
    token_path = corpus_path + ".tokens"
    
    if not os.path.exists(token_path) or os.path.getsize(token_path) == 0:
        logger.info("Tokenizing corpus...")
        
        # Process in chunks for memory efficiency
        tokens = []
        chunk_size = 1000000  # 1MB chunks
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunk_tokens = tokenizer.encode(chunk).ids
                tokens.extend(chunk_tokens)
        
        # Save as memory-mapped file
        np.array(tokens, dtype=np.int32).tofile(token_path)
        logger.info(f"Tokenized corpus saved to {token_path}")
    
    # Load memory-mapped tokens
    mmap = np.memmap(token_path, dtype=np.int32, mode='r')
    logger.info(f"Loaded {len(mmap):,} tokens from {token_path}")
    return mmap

if __name__ == "__main__":
    # Extract text from PDFs and TXT files
    textAll = extract_text_from_pdfs_and_txts()
    if textAll:
        tokenizer = create_tokenizer(textAll)
        token_save(tokenizer, corpus_path=os.path.join(config["forModel"], config["data_corpus"]))
        logger.info(f"Processing complete. Data saved to '{config['forModel']}'")
    else:
        logger.error("No text extracted. Please check your files.")
    input("Press Enter to exit...")
import os
import glob
import PyPDF2
import numpy as np
from config import config
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

def extract_text_from_pdfs_and_txts(folder=None, forModel=None):
    if folder is None:
        folder = config["forData"]
    if forModel is None:
        forModel = config["forModel"]
    
    os.makedirs(folder, exist_ok=True)
    os.makedirs(forModel, exist_ok=True)
    
    textAll = ""
    pdf_path = os.path.join(folder, "*.pdf")
    txt_path = os.path.join(folder, "*.txt")
    data_corpus_path = os.path.join(forModel, config["data_corpus"])
    
    pdf_Files = glob.glob(pdf_path)
    txt_Files = glob.glob(txt_path)
    
    if not pdf_Files:
        print(f"No PDF files found in '{folder}' directory.")
    else:
        print(f"Found {len(pdf_Files)} PDF files in '{folder}' directory.")
        print("Processing PDF files...")
        for filename in pdf_Files:
            i = 1
            try:
                print(f"Proccesing PDF File:{filename}")
                with open(filename, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        print(f"Proccesed the {i}. page")
                        if page_text:
                            textAll += page_text + " "
                        i+= 1
                print(f"Processed PDF file: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
            
    
    if not txt_Files:
        print(f"No TXT files found in '{folder}' directory.")
    else:
        print(f"Found {len(txt_Files)} TXT files in '{folder}' directory.")
        print("Processing TXT files...")
        try:
            for filename in glob.glob(txt_path):
                with open(filename, "r", encoding="utf-8") as f:
                    textAll += f.read() + " "
                print(f"Processed TXT file: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    textAll = textAll.replace("�", "").replace("---", " ").replace("--","").replace("   ", " ").replace("  ", " ")
    if textAll.strip():
        textAll = " ".join(textAll.split())
        
        if not os.path.exists(data_corpus_path):
            print(f"Creating data corpus file at: {data_corpus_path}")
            os.makedirs(os.path.dirname(data_corpus_path), exist_ok=True)
        
        with open(data_corpus_path, "w", encoding="utf-8") as f:
            f.write(textAll)
        return textAll
    else:
        print("No text extracted from the provided files.")
        return ""
    
def create_tokenizer(text, forModel=None):
    
    if forModel is None:
        forModel = config["forModel"]
    os.makedirs(forModel, exist_ok=True)
        
    tokenizer_path = os.path.join(forModel, config["bpe_tokenizer"])
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    if os.path.exists(tokenizer_path):
        tokenizer.from_file(tokenizer_path)
    
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        show_progress=True,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    tokenizer.train_from_iterator([text], trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(tokenizer_path)
    return tokenizer

def token_save(tokenizer, corpus_path):
    token_path = corpus_path + ".tokens"
    corpus_mtime = os.path.getmtime(corpus_path) if os.path.exists(corpus_path) else 0
    
    # Check if we need to re-tokenize
    if (not os.path.exists(token_path) or 
        os.path.getsize(token_path) == 0 or
        (os.path.exists(token_path) and os.path.getmtime(token_path) < corpus_mtime)):
        
        # Read and tokenize corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            tokens = tokenizer.encode(f.read()).ids
        
        # Save tokens as memory-mapped file
        np.array(tokens, dtype=np.int32).tofile(token_path)
    
    # Load tokens from memory-mapped file
    return np.memmap(
        token_path,
        dtype=np.int32,
        mode='r'
    )

if __name__ == "__main__":
    # Extract text from PDFs and TXT files
    textAll = extract_text_from_pdfs_and_txts()
    if textAll:
        tokenizer = create_tokenizer(textAll)
        token_save(tokenizer,corpus_path=os.path.join(config["forModel"], config["data_corpus"]))
        print(f"Text extraction complete. Data saved to '{config['forModel']}/{config['data_corpus']}'.")
    else:
        print("No text extracted. Please check your files.")
    input("Press Enter to exit...")
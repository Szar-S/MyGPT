import os
import glob
import PyPDF2
from gpt import config
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

def extract_text_from_pdfs_and_txts(folder=None, forModel=None):
    if folder is None:
        folder = config["forData"]
    if forModel is None:
        forModel = config["forModel"]
    textAll = ""
    pdf_path = os.path.join(folder, "*.pdf")
    txt_path = os.path.join(folder, "*.txt")
    data_corpus_path = os.path.join(forModel, config["data_corpus"])
    if not os.path.exists(folder):
        print(f"Directory '{folder}' does not exist.")
        os.makedirs(folder)
        print(f"Created directory: {folder}")
        print(f"Please add your PDF or TXT files to the '{folder}' directory.")
        return ""
    if not os.path.exists(forModel):
        os.makedirs(forModel)
    if not glob.glob(pdf_path):
        print(f"No PDF or TXT files found in '{folder}' directory.")
    else:
        for filename in glob.glob(pdf_path):
            with open(filename, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        textAll += page_text + "\n"
                print(f"Processed PDF file: {filename}")
    if not glob.glob(txt_path):
        print(f"No TXT files found in '{folder}' directory.")
    else:
        for filename in glob.glob(txt_path):
            with open(filename, "r", encoding="utf-8") as f:
                textAll += f.read() + "\n"
            print(f"Processed TXT file: {filename}")
            
    textAll = " ".join(textAll.strip())
    if not os.path.exists(data_corpus_path):
        print(f"Creating data corpus file at: {data_corpus_path}")
        os.makedirs(os.path.dirname(data_corpus_path), exist_ok=True)
    with open(data_corpus_path, "w", encoding="utf-8") as f:
        f.write(textAll)
    return textAll
    
def create_tokenizer(text, forModel=None):
    if forModel is None:
        forModel = config["forModel"]
    tokenizer_path = os.path.join(forModel, config["bpe_tokenizer"])
    os.makedirs(forModel, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        show_progress=True,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    tokenizer.train_from_iterator([text], trainer)
    tokenizer.decoder = decoders.BPEDecoder()
    tokenizer.save(tokenizer_path)

if __name__ == "__main__":
    # Extract text from PDFs and TXT files
    textAll = extract_text_from_pdfs_and_txts()
    if textAll:
        create_tokenizer(textAll)
        print(f"Text extraction complete. Data saved to '{config['forModel']}/{config['data_corpus']}'.")
    else:
        print("No text extracted. Please check your files.")
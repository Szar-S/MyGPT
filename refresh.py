import os
import glob
import PyPDF2
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

def extract_text_from_pdfs_and_txts(folder="data", forModel="forModel"):
    textAll = ""
    pdf_path = os.path.join(folder, "*.pdf")
    txt_path = os.path.join(folder, "*.txt")
    data_corpus_path = os.path.join(forModel, "data_corpus.txt")
    if not os.path.exists(folder):
        print("Directory 'data' does not exist.")
        os.makedirs(folder)
        print(f"Created directory: {folder}")
        print("Please add your PDF or TXT files to the 'data' directory.")
        return ""
    if not os.path.exists(forModel):
        os.makedirs(forModel)
    if not glob.glob(pdf_path):
        print("No PDF or TXT files found in 'data' directory.")
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
        print("No TXT files found in 'data' directory.")
    else:
        for filename in glob.glob(txt_path):
            with open(filename, "r", encoding="utf-8") as f:
                textAll += f.read() + "\n"
            print(f"Processed TXT file: {filename}")
    textAll = " ".join(textAll.split())
    if not os.path.exists(data_corpus_path):
        print(f"Creating data corpus file at: {data_corpus_path}")
        os.makedirs(os.path.dirname(data_corpus_path), exist_ok=True)
    with open(data_corpus_path, "w", encoding="utf-8") as f:
        f.write(" ".join(textAll.split()))
    
    return textAll
    
def create_tokenizer(text, forModel="forModel"):
    tokenizer_path = os.path.join(forModel, "bpe_tokenizer.json")
    os.makedirs(forModel, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=2000,
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
        print("Text extraction complete. Data saved to 'forModel/data_corpus.txt'.")
    else:
        print("No text extracted. Please check your files.")
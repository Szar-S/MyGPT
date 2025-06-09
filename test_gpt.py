import unittest
import torch
import tempfile
import os
from gpt import NanoGPT, TextDataset, top_k_top_p_filtering
from refresh import create_tokenizer
from tokenizers import Tokenizer

class TestGPT(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        # Create sample corpus
        self.corpus_path = os.path.join(self.test_dir.name, "corpus.txt")
        with open(self.corpus_path, "w") as f:
            f.write("This is a test document. " * 100)
        
        # Create tokenizer
        self.tokenizer_path = os.path.join(self.test_dir.name, "tokenizer.json")
        create_tokenizer("This is a test sentence. " * 100, forModel=self.test_dir.name)
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

    def test_model_forward(self):
        model = NanoGPT(vocab_size=self.tokenizer.get_vocab_size())
        x = torch.randint(0, 100, (2, 10))
        output = model(x)
        self.assertEqual(output.shape, (2, 10, model.token_embed.num_embeddings))

    def test_streaming_dataset(self):
        dataset = TextDataset(
            file_path=self.corpus_path,
            tokenizer=self.tokenizer,
            seq_len=32
        )
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertEqual(sample["input_ids"].shape, (32,))
        self.assertEqual(sample["labels"].shape, (32,))

    def test_generation(self):
        model = NanoGPT(vocab_size=self.tokenizer.get_vocab_size())
        # Test sampling filter
        logits = torch.randn(100)
        filtered = top_k_top_p_filtering(logits, top_k=10, top_p=0.9)
        self.assertLess(torch.sum(torch.isinf(filtered)), 90)

    def tearDown(self):
        self.test_dir.cleanup()

if __name__ == "__main__":
    unittest.main()
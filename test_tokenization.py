import unittest
from tokenization import NllbTokenizer
from tokenization import ByteTokenizer, SentinelTokenizer
from torch import tensor


class TestTokenization(unittest.TestCase):

    def test_tokenized_mixture_of_bitexts(self):
        with open("test_files/lang1.txt") as reader:
            lines = [line.strip() for line in reader.readlines()]
        tokenizer = NllbTokenizer("600M")
        tokens = tokenizer(lines[:3], lang_code="eng_Latn")
        expected_lang1_token_ids = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 1],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, 1, 1],
            ]
        )
        expected_lang1_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]
        )
        self.assertEqual(
            tokens["input_ids"].tolist(), expected_lang1_token_ids.tolist()
        )
        self.assertEqual(
            tokens["attention_mask"].tolist(), expected_lang1_mask.tolist()
        )

    def test_tokenized_mixture_of_bitexts_truncated(self):
        with open("test_files/lang1.txt") as reader:
            lines = [line.strip() for line in reader.readlines()]
        tokenizer = NllbTokenizer("600M", max_length=8)
        tokens = tokenizer(lines[:3], lang_code="eng_Latn")
        expected_lang1_token_ids = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, 1],
            ]
        )
        expected_lang1_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
        self.assertEqual(
            tokens["input_ids"].tolist(), expected_lang1_token_ids.tolist()
        )
        self.assertEqual(
            tokens["attention_mask"].tolist(), expected_lang1_mask.tolist()
        )

    def test_hf_tokenizer_properties(self):
        tokenizer = NllbTokenizer("1.3B", max_length=8)
        self.assertEqual(len(tokenizer), 256204)
        special_tokens = tokenizer.get_special_tokens()
        self.assertEqual(len(special_tokens), 207)
        self.assertEqual(special_tokens["<s>"], 0)
        self.assertEqual(special_tokens["<pad>"], 1)
        self.assertEqual(special_tokens["</s>"], 2)
        self.assertEqual(special_tokens["<unk>"], 3)
        self.assertEqual(special_tokens["<mask>"], 256203)

    def test_byte_tokenizer1(self):
        tokenizer = ByteTokenizer()
        lines = ["cat"]
        tokenized = tokenizer(lines, lang_code="eng_Latn")
        eng_id = tokenizer.get_special_tokens()["eng_Latn"]
        eos = tokenizer.get_special_tokens()["<eos>"]
        expected = {"input_ids": [[eng_id, 99, 97, 116, eos]]}
        self.assertEqual(tokenized["input_ids"].tolist(), expected["input_ids"])

    def test_byte_tokenizer2(self):
        tokenizer = ByteTokenizer()
        lines = ["cat", "dogs"]
        tokenized = tokenizer(lines)
        eng_id = tokenizer.get_special_tokens()["eng_Latn"]
        eos = tokenizer.get_special_tokens()["<eos>"]
        pad = tokenizer.get_special_tokens()["<pad>"]
        expected = {
            "input_ids": [
                [eng_id, 99, 97, 116, eos, pad],
                [eng_id, 100, 111, 103, 115, eos],
            ]
        }
        self.assertEqual(tokenized["input_ids"].tolist(), expected["input_ids"])

    def test_byte_tokenizer3(self):
        tokenizer = ByteTokenizer()
        lines = ["cat", "안녕"]
        tokenized = tokenizer(lines)
        eng_id = tokenizer.get_special_tokens()["eng_Latn"]
        eos = tokenizer.get_special_tokens()["<eos>"]
        pad = tokenizer.get_special_tokens()["<pad>"]
        expected = {
            "input_ids": [
                [eng_id, 99, 97, 116, eos, pad, pad, pad],
                [eng_id, 236, 149, 136, 235, 133, 149, eos],
            ]
        }
        self.assertEqual(tokenized["input_ids"].tolist(), expected["input_ids"])

    def test_byte_tokenizer4(self):
        tokenizer = ByteTokenizer()
        lines = ["cat", "dogs"]
        tokenized = tokenizer(lines, lang_code="fra_Latn")
        lang_id = tokenizer.get_special_tokens()["fra_Latn"]
        eos = tokenizer.get_special_tokens()["<eos>"]
        pad = tokenizer.get_special_tokens()["<pad>"]
        expected = {
            "input_ids": [
                [lang_id, 99, 97, 116, eos, pad],
                [lang_id, 100, 111, 103, 115, eos],
            ]
        }
        self.assertEqual(tokenized["input_ids"].tolist(), expected["input_ids"])

    def test_byte_tokenizer5(self):
        tokenizer = ByteTokenizer(offset=100)
        lines = ["cat", "dogs"]
        tokenized = tokenizer(lines, lang_code="fra_Latn")
        lang_id = tokenizer.get_special_tokens()["fra_Latn"]
        eos = tokenizer.get_special_tokens()["<eos>"]
        pad = tokenizer.get_special_tokens()["<pad>"]
        expected = {
            "input_ids": [
                [lang_id, 199, 197, 216, eos, pad],
                [lang_id, 200, 211, 203, 215, eos],
            ]
        }
        self.assertEqual(tokenized["input_ids"].tolist(), expected["input_ids"])

    def test_byte_tokenizer6(self):
        tokenizer = SentinelTokenizer(offset=0)
        lines = ["c😀a😀t😀", "dog😀s"]
        tokenized = tokenizer(lines, lang_code="fra_Latn")
        lang_id = tokenizer.get_special_tokens()["fra_Latn"]
        eos = tokenizer.get_special_tokens()["<eos>"]
        pad = tokenizer.get_special_tokens()["<pad>"]
        print(tokenized["input_ids"].tolist())


if __name__ == "__main__":
    unittest.main()

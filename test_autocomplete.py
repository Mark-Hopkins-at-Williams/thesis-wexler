from autocomplete import compress
import unittest
from torch import tensor
import torch


class MockAutocompleteModel:
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def eval(self):
        pass

    def __call__(self, input_ids):
        preds = (input_ids + 1) % len(self.alphabet)
        logits = torch.zeros(preds.shape[0], preds.shape[1], len(self.alphabet))
        logits = logits.scatter(dim=2, index=preds.unsqueeze(-1), value=1)
        return logits


def strings_to_ids(input_strings, letter_to_id):
    rows = []
    for input_string in input_strings:
        row = tensor([letter_to_id[letter] for letter in input_string])
        rows.append(row)
    return torch.stack(rows)


class TestAutocomplete(unittest.TestCase):

    def test_compress(self):
        alphabet = "abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        input_strings = ["abcabc", "bcabca"]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)

        # target_ids = (input_ids + 1) % len(alphabet)
        target_strings = ["bcacca", "cbbcab"]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(model, input_ids, target_ids, output_style="-short")
        print(text)


if __name__ == "__main__":
    unittest.main()

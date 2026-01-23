from autocomplete import compress
import unittest
from torch import tensor
import torch
import re


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
        input_strings = ["abcabc", "bcabca", "abcabc", "abcabc", "abcabc", "accccc", "ccccca", "ccccca"]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)
        print(f"Inputs                  = {input_strings}")
        print(f"Preds                   = {self.logits_to_strings(model(input_ids))}")

        target_strings = ["bcacca", "cbbcab", "bcabca", "abcabc", "bbabba", "bbbbbb", "bbbbbb", "baaaab"]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(model, input_ids, target_ids, output_style="-long")
        #print(f"Condensed rep = {text}")
        print(f"Condensed rep (decoded) = {self.decode_hex_list(text)}")

        """
        abc -> bca is perfect
        abc -> abc is 100% wrong
        abc -> bba is wrong right wrong
        """

    def decode_hex_list(self, input_list):
      decoded_result = []
      for line in input_list:
          cleaned = line.replace(r"\x00", "a").replace(r"\x01", "b").replace(r"\x02", "c")
          
          parts = re.split(r"(😀)", cleaned)
          
          # Filter out empty strings from the split
          filtered_parts = [p for p in parts if p]
          
          if filtered_parts:
              decoded_result.append("".join(filtered_parts))
              
      return decoded_result


    def logits_to_strings(self, logits_tensor):
      alphabet = "abc"
      
      # Find the index of the max value (1.0) along the last dimension
      indices = torch.argmax(logits_tensor, dim=2)
      
      decoded_strings = []
      for row in indices:
          # Convert indices back to characters using the string index
          text = "".join([alphabet[i] for i in row])
          decoded_strings.append(text)
          
      return decoded_strings


if __name__ == "__main__":
    unittest.main()

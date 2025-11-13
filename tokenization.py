import sys
from transformers import AutoTokenizer
from typing import Dict, Tuple, List, Optional, Iterator, Callable
import warnings
from abc import ABC
from abc import abstractmethod
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding



class Tokenizer(ABC):
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __call__(self, sents: List[str]):  
        pass
    
    @abstractmethod
    def get_special_tokens(self):
        pass
    
    @abstractmethod
    def batch_decode(self):
        pass


class CharacterTokenizer(Tokenizer):

    def __init__(self, max_length=None):
      # define full vocabulary for eng-fra
      self.beg_token = "<bos>"
      self.end_token = "<eos>"
      self.pad_token = "<pad>"
      self.mask_token = "<mask>"
      self.unk_token = "<unk>"
      self.special_tokens = [self.beg_token, self.end_token, self.unk_token, self.pad_token, self.mask_token, "eng_Latn", "fra_Latn"]
      self.vocab_str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^îâéïüøàèì{|}~ "
      self.vocab = list(self.vocab_str)

      # the characters that appeared over 500 times in the 18 million examples dataset
      top_chars = ['ў', 'έ', 'ъ', 'ή', 'К', 'Ï', 'Т', 'ґ', 'Њ', 'ū', 'ī', '\x91', 'э', '‡', 'Љ', 'ţ', 'Ε', 'М', 'đ', 'А', 'ه', 'П', 'ά', 'ό', 'і', 'ك', 'ÿ', '¦', 'ف', '\x99', 'س', '¡', 'ة', '¢', '‹', '¹', 'ő', '›', '□', 'ς', 'щ', 'Ø', 'Ñ', '●', 'ί', 'ع', 'Ş', 'δ', 'د', 'θ', 'ф', '†', 'υ', '¬', '́', 'ř', 'ě', 'ب', '‚', '\x95', 'Ќ', 'İ', '′', 'ª', 'η', '³', 'Ѓ', 'ż', 'κ', 'Í', 'ш', 'ю', 'Є', 'ت', 'β', 'ð', 'π', 'Ä', '→', 'Ó', '‰', 'Ο', 'γ', 'ą', 'Ћ', '¶', 'ц', 'Ž', 'ś', 'و', 'Å', 'ن', 'ν', 'σ', 'õ', 'ė', 'ì', 'Δ', 'ر', 'م', 'ي', 'ğ', 'ā', 'λ', '\u200e', 'ý', 'ρ', 'Œ', 'ę', '^', 'В', 'Ü', 'ă', 'х', '\x94', 'ж', '\x9c', '≥', '×', 'ń', '½', 'Р', '\x96', '\x93', '¨', 'ο', 'μ', 'τ', 'ι', 'Ђ', 'Č', 'Ô', 'ل', 'ч', 'ε', '√', 'æ', 'µ', '~', '\uf03d', 'ş', '‑', '„', '¿', 'Ö', 'ı', 'ь', 'ł', 'б', 'Ё', '\x97', 'ا', 'ò', 'ž', '≤', 'г', 'з', 'ы', 'Â', '£', 'Á', 'ß', 'я', 'Ê', 'α', 'Ç', 'å', '§', 'у', 'п', 'Š', 'д', '·', 'м', '\\', 'ø', '\xad', 'й', 'к', 'л', '∗', 'č', 'º', 'в', 'š', '}', '{', 'с', '™', 'р', 'С', 'Î', '±', 'ú', 'т', 'н', 'ã', 'и', '#', 'е', '²', 'Ã', 'ć', 'а', 'È', 'ñ', 'о', '‘', '®', 'Г', '€', 'ä', '<', '©', '´', '\x92', '`', 'ü', 'ö', '>', '+', '@', 'ó', 'í', '|', '°', 'á', '=', '−', '&', 'ë', '�', '–', '•', '—', '_', '*', '…', 'ï', '$', 'Z', 'ù', '!', 'À', 'œ', ']', 'X', 'û', '[', 'î', 'â', '?', '%', 'Q', 'ç', 'Y', '»', '«', '”', '“', '"', 'ô', 'K', 'É', 'V', 'J', '’', 'W', 'ê', ':', 'H', ';', '8', '7', 'B', '6', 'F', '4', 'z', '5', 'G', '3', 'O', '9', '/', 'U', 'D', 'R', '\xa0', '(', 'è', 'M', 'N', 'j', 'E', ')', 'L', 'k', 'P', 'T', '-', 'à', 'I', '1', '2', 'S', 'A', '0', 'x', 'C', 'q', 'w', "'", 'y', '\n', '.', 'b', ',', 'v', 'g', 'é', 'f', 'h', 'p', 'm', 'c', 'd', 'u', 'l', 'r', 'o', 's', 'a', 'i', 'n', 't', 'e', ' ']

      # add frequent characters I may have forgotten in vocab_str
      for char in top_chars:
        if char not in self.vocab:
          self.vocab.append(char)
      
      self.vocab = self.special_tokens + self.vocab 

      self.max_length = max_length

      # assign and store ID for each token (+ reverse)
      self.mappings = {} # char -> ID
      self.reverse_mappings = {} # ID -> char
      i = 0
      for char in self.vocab:
        self.mappings[char] = i
        i = i+1 
      self.reverse_mappings = {v: k for k, v in self.mappings.items()}

    def __len__(self):
        return len(self.vocab)
    
    def __call__(self, sents: List[str], lang_code=None):
      if lang_code is not None:
        self.src_lang = lang_code

      encoded = []

      for sentence in sents:
          # Convert each char in a sentence to its mapped ID (using <UNK> if not found)
          char_ids = [self.mappings.get(char, self.mappings[self.unk_token]) for char in sentence]
          # Build the full sequence as a tensor
          ids = torch.tensor(
              [self.mappings["eng_Latn"]] + char_ids + [self.mappings[self.end_token]],
              dtype=torch.long
          )
          if self.max_length != None and ids.size(0) > self.max_length: # truncate sequences that are too long
            ids = ids[:self.max_length]
            ids[self.max_length-1] = self.mappings[self.end_token] # insert EOS token in truncated sequences
          encoded.append(ids)

      # Pad all sequences to the same length
      input_ids = pad_sequence(
          encoded, batch_first=True, padding_value=self.mappings[self.pad_token]
      )
      # Create attention mask (1 where token is not PAD)
      attention_mask = (input_ids != self.mappings[self.pad_token]).long()

      return BatchEncoding(
        {
          "input_ids": input_ids,
          "attention_mask": attention_mask
        },
        tensor_type="pt"
      )
   
    
    def get_special_tokens(self):
      return {tok : self.mappings[tok] for tok in self.special_tokens}
      
    
    
    def batch_decode(self, token_ids):
        results = [] # list of strings, each of which is decoded sentence

        non_printables = set(self.special_tokens)
        non_printables.remove(self.unk_token) # we want to see the unk token when printing

        for sent in token_ids:
            # Convert all token IDs in one go
            decoded_tokens = [self.reverse_mappings[id.item()] for id in sent]

            # Filter out special tokens and make string representation
            decoded = ''.join(tok for tok in decoded_tokens if tok not in non_printables)
            results.append(decoded)

        return results

class HuggingfaceTokenizer(Tokenizer):
    
    def __init__(self, model_name, max_length=None):
        self.max_length = max_length        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`clean_up_tokenization_spaces` was not set.*",
                category=FutureWarning,
                module="transformers.tokenization_utils_base",
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except OSError:
                sys.stderr.write('Tokenizer not found. Using NLLB tokenizer instead.\n')
                sys.stderr.flush()
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.special_tokens = dict(zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids))
        
    def __len__(self):
        return len(self.tokenizer)
    
    def __call__(self, sents: List[str], lang_code=None):        
        if lang_code is not None:
            self.tokenizer.src_lang = lang_code
        return self.tokenizer(
            sents, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length if self.max_length is not None else None
        )
        
    def get_special_tokens(self):
        return self.special_tokens
    
    def batch_decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    
class NllbTokenizer(HuggingfaceTokenizer):
    def __init__(self, size, max_length=None):
        super().__init__(f"facebook/nllb-200-distilled-{size}", max_length=max_length)
        
def main():
  x = CharacterTokenizer()
  # strs = ["The quick brown fox jumps over the lazy dog", "I Love ToK€NIZation!?!"]
  strs = ["Tokens", "Tok€ns"]
  tokenized = x(strs)
  print(tokenized)
  print(x.get_special_tokens())
  print(x.batch_decode(tokenized['input_ids']))

  print(x.vocab)

  # x = CharacterTokenizer(max_length=8)
  # strs = ["The quick brown fox jumps over the lazy dog", "I Love ToKéNIZation!?!"]
  # tokenized = x(strs)
  # print(tokenized)
  # print(x.get_special_tokens())
  # print(x.batch_decode(tokenized['input_ids']))


  # x = NllbTokenizer("600M", max_length=8)
  # print(x(strs))
  # print(x.get_special_tokens())
  # print(x.batch_decode(x(strs)["input_ids"]))


if __name__ == "__main__":
    main()



""" First try, worse code """
   # def __call__(self, sents: List[str]):
    #   # create encodings 
    #   encoded = []
    #   longest_len = 0  

    #   for sentence in sents:
    #     ids = []
    #     ids.append(self.mappings["eng_Latn"]) # prepend with lang code

    #     # add each char's encoding
    #     for char in sentence:
    #       # if we hit a number higher than max_len, break
    #       if char in self.vocab:
    #         ids.append(self.mappings[char]) # add char's ID
    #       else:
    #         ids.append(self.mappings["<UNK>"]) # add ID for unknown character

    #     ids.append(self.mappings["<EOS>"]) # end of sentence
    #     encoded.append(ids)

    #     # determine which encoding is longest
    #     if len(ids) > longest_len:
    #       longest_len = len(ids)


    #   # make all vectors the same length + create attention mask
    #   attention_mask = []

    #   for encoding in encoded:
    #     # mark the tokens model should attend to 
    #     sentence_mask = len(encoding) * [1]

    #     # handle padding in input_ids and mask
    #     if len(encoding) < longest_len:
    #       len_difference = longest_len - len(encoding)
    #       encoding += len_difference * [self.mappings["<PAD>"]] # add pad tokens to encoded representation
    #       sentence_mask += len_difference * [0] # add pad tokens to attention_mask

    #     attention_mask.append(sentence_mask)
          
    #   return {
    #     'input_ids': encoded, 
    #     'attention_mask': attention_mask
    #     } 


    # def batch_decode(self, token_ids):
      # for sent in token_ids:
          #   decoded_sent = ""
          #   for curr_id in sent:
          #     decoded_token = self.reverse_mappings[curr_id.item()] # turn token ID into token char
          #     if decoded_token not in self.special_tokens: # we don't want to see EOS and things
          #       decoded_sent = decoded_sent + decoded_token
          #   results.append(decoded_sent)

        # return results 

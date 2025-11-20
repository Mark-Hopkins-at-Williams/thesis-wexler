import json
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from configure import USE_CUDA
from corpora import Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
from validate import translate_tokenized_mixture_of_bitexts, translate
from tokenization import NllbTokenizer, HuggingfaceTokenizer, CharacterTokenizer

# parameters
model_dir = "/mnt/storage/swexler/thesis-wexler/models/french-training-v23"

model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
if USE_CUDA:
    model.cuda()

text_files = {
    ("test", "eng"): "/mnt/storage/swexler/thesis-wexler/examples/french-model_10_28_25/fr-en/test.eng",
    ("test", "fra"): "/mnt/storage/swexler/thesis-wexler/examples/french-model_10_28_25/fr-en/test.fr",
}
lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
mix = MixtureOfBitexts.create_from_files(
    text_files,
    [(("test", "eng"), ("test", "fra"), None)],
    batch_size=1,
    only_once_thru=True,
)


tgt_tokenizer = NllbTokenizer("600M")
src_tokenizer = CharacterTokenizer(offset=len(tgt_tokenizer), max_length=128)
mix = TokenizedMixtureOfBitexts(mix, src_tokenizer, tgt_tokenizer, lang_codes)
pmap = dict()


batch = mix.next_batch()  
translations = dict()
while batch is not None:
    src, _, src_lang, tgt_lang = batch        
    permutation = pmap[tgt_lang] if tgt_lang in pmap else None
    src_code = lang_codes[src_lang]
    tgt_code = lang_codes[tgt_lang]
    key = '->'.join([src_code, tgt_code])
    if key not in translations:
        translations[key] = []
    translated = translate(src, tgt_tokenizer, model, tgt_code, permutation) # tgt lang's tokenizer
    print(translated)
    
    translations[key].extend(translated)
    batch = mix.next_batch() 


#translations = translate_tokenized_mixture_of_bitexts(
#    tmob, model, tgt_tokenizer, lang_codes
#)

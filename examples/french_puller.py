from datasets import load_dataset
from tqdm import tqdm
import os

# OVERALL SOURCE: https://huggingface.co/datasets/wmt/wmt14/viewer/fr-en
# what langs from HF do we want to pull
lang_pairs = ["fr-en"]

# put all lang data in one data structure
all_datasets = {}
for curr_lang_pair in lang_pairs:
  all_datasets[curr_lang_pair] = load_dataset("wmt/wmt14", curr_lang_pair) 


# create the data files
for lang_pair, lang_pair_data in all_datasets.items():
  # make a folder for the languge pair and save name of creole variant
  path = "/mnt/storage/swexler/thesis-wexler/examples/french-model_10_28_25/" + lang_pair + "/"
  os.makedirs(path)
  lang_code = lang_pair.split("-")[0]

  # define where french source sentences will be stored
  train_file_fr = path + "train." + lang_code
  dev_file_fr = path + "dev." + lang_code
  test_file_fr = path + "test." + lang_code

  # define where english translations will be stored
  train_file_eng = path + "train.eng"
  dev_file_eng = path + "dev.eng"
  test_file_eng = path + "test.eng"



  # WRITE TO FILES
  line_num = 0 # to determine end bounds

  # training examples writing
  with open(train_file_fr, "w") as file_fr, open(train_file_eng, "w") as file_eng:
    for example in tqdm(lang_pair_data['train']): 
      file_fr.write(example['translation']['fr'] + "\n")
      file_eng.write(example['translation']['en'] + "\n")
      line_num = line_num+1
    print("last written line in " , lang_code, " train file =", line_num-1)
      
  # dev examples writing
  with open(dev_file_fr, "w") as file_fr, open(dev_file_eng, "w") as file_eng:
    for example in tqdm(lang_pair_data['validation']): 
      file_fr.write(example['translation']['fr'] + "\n")
      file_eng.write(example['translation']['en'] + "\n")
      
  # test examples writing
  with open(test_file_fr, "w") as file_fr, open(test_file_eng, "w") as file_eng:
    for example in tqdm(lang_pair_data['test']): 
      file_fr.write(example['translation']['fr'] + "\n")
      file_eng.write(example['translation']['en'] + "\n")
      
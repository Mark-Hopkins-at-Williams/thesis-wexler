from collections import defaultdict

def read_eng(data_path):
  with open(data_path+"/train.eng", "r") as f:
    eng_file = f.readlines()
  return eng_file

def read_fra(data_path):
  with open(data_path+"/train.fr", "r") as f:
    fr_file = f.readlines()
  return fr_file


def modify_eng(eng_file):
  eng_file.pop(18155213)
  return eng_file

def modify_fra(fr_file):
  fr_concat = [
    [18157028, 18157029, 18157030],
    [18157033, 18157034, 18157035],
    [18157041, 18157042, 18157043],
    [18157045, 18157046, 18157047]
  ]

  # combining seperate sentences in fr file that are together in eng file
  for issue in reversed(fr_concat): # working through the issues in backwards order to not mess up indexing
    fr_file[issue[0]] += fr_file[issue[1]] + fr_file[issue[2]]
    fr_file.pop(issue[2])
    fr_file.pop(issue[1])
  return fr_file


def getlines(fr_file, eng_file, start, stop, step):
  for iter in range(2):
    i = start
    if iter == 0: 
      print("------ENG------")
      file_selected = eng_file[start:stop:step]
      for sentence in file_selected:
        print(str(i) + ": " + sentence + "\n")
        i = i+step
    else:
      print("------FR------")
      file_selected = fr_file[start:stop:step]
      for sentence in file_selected:
        print(str(i) + ": " + sentence + "\n")
        i = i+step
    
    
def main():
  # data_path = "examples/french-model_10_28_25/fr-en"
  data_path = "examples/french-data-18-mil"

  print("reading Eng file")
  eng_file_list = read_eng(data_path) # the English file (represented as a list)

  print("reading Fra file")
  fra_file_list = read_fra(data_path) # the French file (represented as a list)

  """
  eng_file_list_modified = modify_eng(eng_file_list) # Eng file modified to be parallel (attempted)
  fra_file_list_modified = modify_fra(fra_file_list) # Fra file modified to be parallel (attempted)

  start = 18207010
  stop = 18208010
  step = 20

  getlines(fra_file_list_modified, eng_file_list_modified, start, stop, step)

  # print(len(eng_file_list))
  # print(len(fra_file_list))
  # print(eng_file_list[-1])
  # print(fra_file_list[-1])
  """


  counts = defaultdict(int)
  print("Starting Eng counting")
  for sentence in eng_file_list:
    for char in sentence:
      counts[char] += 1
  print("Starting Fra counting")
  for sentence in fra_file_list:
    for char in sentence:
      counts[char] += 1

  counts = dict(sorted(counts.items(), key=lambda item: item[1]))

  print(counts)

  # benchmark = 500
  # filtered_dict = {key: value for key, value in counts.items() if value >= benchmark}
  # print(list(filtered_dict.keys()))

  top_chars = ['ў', 'έ', 'ъ', 'ή', 'К', 'Ï', 'Т', 'ґ', 'Њ', 'ū', 'ī', '\x91', 'э', '‡', 'Љ', 'ţ', 'Ε', 'М', 'đ', 'А', 'ه', 'П', 'ά', 'ό', 'і', 'ك', 'ÿ', '¦', 'ف', '\x99', 'س', '¡', 'ة', '¢', '‹', '¹', 'ő', '›', '□', 'ς', 'щ', 'Ø', 'Ñ', '●', 'ί', 'ع', 'Ş', 'δ', 'د', 'θ', 'ф', '†', 'υ', '¬', '́', 'ř', 'ě', 'ب', '‚', '\x95', 'Ќ', 'İ', '′', 'ª', 'η', '³', 'Ѓ', 'ż', 'κ', 'Í', 'ш', 'ю', 'Є', 'ت', 'β', 'ð', 'π', 'Ä', '→', 'Ó', '‰', 'Ο', 'γ', 'ą', 'Ћ', '¶', 'ц', 'Ž', 'ś', 'و', 'Å', 'ن', 'ν', 'σ', 'õ', 'ė', 'ì', 'Δ', 'ر', 'م', 'ي', 'ğ', 'ā', 'λ', '\u200e', 'ý', 'ρ', 'Œ', 'ę', '^', 'В', 'Ü', 'ă', 'х', '\x94', 'ж', '\x9c', '≥', '×', 'ń', '½', 'Р', '\x96', '\x93', '¨', 'ο', 'μ', 'τ', 'ι', 'Ђ', 'Č', 'Ô', 'ل', 'ч', 'ε', '√', 'æ', 'µ', '~', '\uf03d', 'ş', '‑', '„', '¿', 'Ö', 'ı', 'ь', 'ł', 'б', 'Ё', '\x97', 'ا', 'ò', 'ž', '≤', 'г', 'з', 'ы', 'Â', '£', 'Á', 'ß', 'я', 'Ê', 'α', 'Ç', 'å', '§', 'у', 'п', 'Š', 'д', '·', 'м', '\\', 'ø', '\xad', 'й', 'к', 'л', '∗', 'č', 'º', 'в', 'š', '}', '{', 'с', '™', 'р', 'С', 'Î', '±', 'ú', 'т', 'н', 'ã', 'и', '#', 'е', '²', 'Ã', 'ć', 'а', 'È', 'ñ', 'о', '‘', '®', 'Г', '€', 'ä', '<', '©', '´', '\x92', '`', 'ü', 'ö', '>', '+', '@', 'ó', 'í', '|', '°', 'á', '=', '−', '&', 'ë', '�', '–', '•', '—', '_', '*', '…', 'ï', '$', 'Z', 'ù', '!', 'À', 'œ', ']', 'X', 'û', '[', 'î', 'â', '?', '%', 'Q', 'ç', 'Y', '»', '«', '”', '“', '"', 'ô', 'K', 'É', 'V', 'J', '’', 'W', 'ê', ':', 'H', ';', '8', '7', 'B', '6', 'F', '4', 'z', '5', 'G', '3', 'O', '9', '/', 'U', 'D', 'R', '\xa0', '(', 'è', 'M', 'N', 'j', 'E', ')', 'L', 'k', 'P', 'T', '-', 'à', 'I', '1', '2', 'S', 'A', '0', 'x', 'C', 'q', 'w', "'", 'y', '\n', '.', 'b', ',', 'v', 'g', 'é', 'f', 'h', 'p', 'm', 'c', 'd', 'u', 'l', 'r', 'o', 's', 'a', 'i', 'n', 't', 'e', ' ']

if __name__ == "__main__":
    main()
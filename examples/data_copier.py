from tqdm import tqdm

files = ["train.fr", "train.eng"]
source_data_path = "examples/french-model_10_28_25/fr-en/"
new_data_path = "examples/french-data-18-mil/"

num_lines = 18155210

penalty = 0

for file in files:
  with open(source_data_path+file, "r") as src, open(new_data_path+file, "w") as dest:
    # actually transfer text
    for i in range(num_lines):
      dest.write(src.readline())

# see if files are parallel
with open("examples/french-data-18-mil/train.eng", "r") as f:
  a = f.readlines()
with open("examples/french-data-18-mil/train.fr", "r") as g:
  b = g.readlines()
print(len(a))
print(len(b))
print(a[-1])
print(b[-1])


# maybe make a penalty int that is -8 for french and -0 for eng so the files are the same length
# print the last ~200 setnences of each to see what's happening
# but make sure meanings are still parallel


"""
src_list = src.readlines()

    # modify files to make them parallel
    if file == "train.eng":
      print("processing Eng")
      penalty = 0
      src_list.pop(18155213)
    if file == "train.fr":
      print("processing Fr")
      fr_concat = [
        [18157028, 18157029, 18157030],
        [18157033, 18157034, 18157035],
        [18157041, 18157042, 18157043],
        [18157045, 18157046, 18157047]
      ]
      for issue in reversed(fr_concat): # working through the issues in backwards order to not mess up indexing
        src_list[issue[0]] += src_list[issue[1]] + src_list[issue[2]]
        src_list.pop(issue[2])
        src_list.pop(issue[1])

      penalty = 8

    # actually transfer text
    for i in range(num_lines+penalty):
      dest.write(src_list[i])
"""
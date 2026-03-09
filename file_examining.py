import re
from itertools import zip_longest
from tqdm import tqdm

def copy_first_lines(source_filename, destination_filename, num_lines):
    """
    Reads the first num_lines lines of a source file and writes them to a 
    destination file.
    """
    try:
        # Open the source file for reading
        with open(source_filename, 'r') as source_file:
            # Read the first 10 lines
            lines = [source_file.readline() for _ in range(num_lines)]
        
        # Open the destination file for writing
        with open(destination_filename, 'w') as dest_file:
            # Write the read lines to the new file
            dest_file.writelines(lines)
        
        print(f"Successfully copied the first {num_lines} lines from '{source_filename}' to '{destination_filename}'.")

    except FileNotFoundError:
        print(f"Error: The file '{source_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def remove_vowels_from_file(input_filename, output_filename):
    vowels = "aeiouAEIOU"
    translation_table = str.maketrans('', '', vowels) # first 2 params are what to swap (nothing), 3rd param is what to remove

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                for line in tqdm(infile):
                    outfile.write(line.translate(translation_table))
        
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied when accessing the files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_specific_line(filename, line_number):
    # Line numbers in files are typically 1-based, but Python lists are 0-based.
    # We use 'start=1' in enumerate to align the counter with the file line numbers.
    # Alternatively, you can adjust the input number (e.g., line_number - 1).
    target_line_index = line_number - 1

    try:
        with open(filename, 'r') as file:
            for index, line in enumerate(file):
                if index == target_line_index:
                    # Use end='' to prevent an extra newline character from being printed
                    # because the 'line' variable already includes one.
                    # print(line, end='')
                    return line

        print(f"Error: Line {line_number} not found (file ended).")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")




def decode_hex_array(line):
    pattern = r"(\\x[0-9a-fA-F]{2}|.)"
    parts = re.split(r"(😀)", line) # Split by smiley, but keep it in the list
    
    decoded_result = []
    for part in parts:
        if part == "😀":
            decoded_result.append("😀")
        else:
            # This part is a string of hex like \xc3\xa9
            # We convert it to raw bytes and decode
            sub_pattern = r"\\x([0-9a-fA-F]{2})|(.)"
            temp_bytes = bytearray()
            for m in re.finditer(sub_pattern, part):
                h, c = m.groups()
                if h: temp_bytes.append(int(h, 16)) # h = matches a hex byte
                elif c: temp_bytes.extend(c.encode('utf-8')) # c = matches a char
            
            decoded_text = temp_bytes.decode('utf-8', errors='replace')
            if decoded_text:
                decoded_result.append(decoded_text)
    
    return decoded_result

def count_lines(filename):
  with open(filename, 'r') as file:
    line_count = sum(1 for line in file)
  return line_count


def compare_files(written_path, source_path):
    with open(written_path, 'r', encoding='utf-8') as f_written, \
         open(source_path, 'r', encoding='utf-8') as f_source:
        
        # zip_longest ensures we catch if one file is longer than the other
        for line_num, (line_w, line_s) in enumerate(zip_longest(f_written, f_source), start=1):

          
            
            # Check if one file ended early
            if line_w is None:
                print(f"Line {line_num}: Source file is longer than written file.")
                break
            if line_s is None:
                print(f"Line {line_num}: Written file is longer than source file.")
                break

            # Process and Clean
            decoded = "".join(decode_hex_array(line_w)).strip()
            source = line_s.strip()
            source = source[:1024]

            if line_num % 100000 == 0:
              print(f"checking line {line_num}")

           # FILTER: Only flag if it's NOT a truncation
            if not source.startswith(decoded):
                print("=========")
                print(f"Line {line_num} is a 'mismatch':")
                # print(f"   Decoded: {decoded}")
                # print()
                # print(f"   Source:  {source}")

############################## DRIVER CODE ##############################


# source_file_name = 'examples/english-data-compressed_1_11_26/compressed-train.eng'
# destination_file_name = 'examples/english-data-compressed_1_11_26/compressed-train-7m.eng'
# copy_first_lines(source_file_name, destination_file_name, 18155210)

file_comp_short = "examples/english-data-compressed_1_11_26/compressed-train-short.eng"
file_comp_long = "models/autocomplete-v2/compressed-train.eng"
file_orig = "examples/french-data-7-mil/organized/train.eng"
french_file_orig = "examples/french-data-7-mil/organized/train.fr"
files = [file_comp_short, file_comp_long, file_orig, french_file_orig]


# print(count_lines(file_comp_short))
# print(count_lines(file_comp_long))
# print(count_lines(file_orig))
# print(count_lines(french_file_orig))

fr_file = "examples/french-data-7-mil-512-filtered/train.fr"
eng_file = "examples/french-data-7-mil-512-filtered/train.eng"

# for line in range(5500000, 5500011):
#   print(f"-------LINE {line}-------")
#   print(get_specific_line(fr_file, line))
#   print(get_specific_line(eng_file, line))
#   print("====SHORT CONDENSED====")
#   print("".join(decode_hex_array(get_specific_line(file_comp_short, line))))
#   print("====LONG CONDENSED====")
#   print("".join(decode_hex_array(get_specific_line(file_comp_long, line))))
#   print()

# longest_len = 0
# with open(eng_file, "r") as file:
#   for line in tqdm(file):
#     line_len = len([byte for byte in line.encode()])
#     if line_len > longest_len:
#       longest_len = line_len
# print(longest_len)

  



# file_path = "examples/one-char-examining/compressed-dummy-long.eng"
# for line_num in range(1, count_lines(file_path)+1):
#   print("".join(decode_hex_array(get_specific_line(file_path, line_num))))

# compare_files(written_file, source_file)

# ex_file = "examples/english-data-compressed_1_22_26/compressed-train-short.eng"
# ex_file_organized = "examples/english-data-compressed_1_22_26/organized/compressed-train-short.eng"

# count = 0
# with open(ex_file_organized, "r") as file:
#   for line in tqdm(file):
#     count += 1
# print(count)

# count = 0
# with open(ex_file, "r") as file:
#   for line in tqdm(file):
#     if len(line) < 512:
#       count += 1
# print(count)


# source_file = "examples/french-data-7-mil-512-filtered/dev.eng"
# written_file = "examples/english-data-compressed_1_28_26-0.5/compressed-dev-short.eng"
# old_line = get_specific_line(source_file, 31)
# new_line = "".join(decode_hex_array(get_specific_line(written_file, 31)))
# new_line_str = get_specific_line(written_file, 31)
# new_line_arr = decode_hex_array(get_specific_line(written_file, 31))

# reading_file = "examples/english-data-compressed_2_23_26-0.2-margin/compressed-dev-short.eng"
# lens = []
# with open(reading_file, "r") as file:
#   for line in tqdm(file):
#     lens.append(len("".join(decode_hex_array(line.strip()))))
# print(lens)
# print(sum(lens) / len(lens))

with open("examples/one-char-examining/one-char.eng", 'r') as file:
    line_count = sum(1 for line in file)

for line in range(1, line_count+1):
  print(f"-------LINE {line}-------")
  print(get_specific_line("examples/one-char-examining/one-char.eng", line).strip())
  print("====LONG VERSION====")
  print("".join(decode_hex_array(get_specific_line("examples/one-char-examining/compressed-dummy-long.eng", line))))
  print("====AUTOCOMPLETED====")
  print("".join(decode_hex_array(get_specific_line("examples/one-char-examining/compressed-dummy-short.eng", line))))
  print()


# with open("examples/one-char-examining/compressed-dummy-short.eng", "r") as file:
#   for line in tqdm(file):
#     print("".join(decode_hex_array(line.strip())))

# print("".join(decode_hex_array(get_specific_line(written_file, 6315717))))

# infile = "examples/french-data-7-mil-512-filtered/train.eng"
# outfile = "examples/no-vowels_french-data-7-mil-512-filtered/train.eng"

# remove_vowels_from_file(infile, outfile)
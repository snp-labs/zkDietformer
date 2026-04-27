import os

input_file = "bookcorpus.csv"  # orignal file path
output_folder = "."              # output path of the generated txt file

# read the file
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# remove it if the first line is '0'
if lines[0].strip() == "0":
    lines = lines[1:]

# split in half and save
mid = len(lines) // 2
chunks = [lines[:mid], lines[mid:]]

for idx, chunk in enumerate(chunks, start=1):
    out_file = os.path.join(output_folder, f"books_large_p{idx}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for line in chunk:
            f.write(line.strip() + "\n")

print(f"Saved {len(lines)} lines into 2 files: books_small_p1.txt, books_small_p2.txt")

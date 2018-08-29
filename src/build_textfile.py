import string
import os
from collections import defaultdict

def load_doc(fname):

    if not os.path.exists(fname):
        print('Invalid path')
        return None

    # loads document with text descriptions into list of strings
    with open(fname, 'r') as file:
        text = file.read()

    return text

def load_descriptions(doc):
    # read descriptions and store them in a dictionary
    desc = defaultdict(list)

    for line in doc.split('\n'):

        tokens = line.split()
        if len(line) < 2:
            continue

        img_id, img_desc = tokens[0], tokens[1:]
        img_id = img_id.split('.')[0]   # get rid of format
        desc[img_id].append(' '.join(img_desc)) # join returns list "img_desc" as a string separated by space

    return desc

def clean_text(desc):
    # Convert all words to lowercase.
    # Remove all punctuation.
    # Remove all words that are one character or less in length (e.g. ‘a’).
    # Remove all words with numbers in them.

    table = str.maketrans('', '', string.punctuation) # make a translation table
    for id, desc_list in desc.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]

            desc = desc.split()                         # tokenize
            desc = [w.lower() for w in desc]            # lowercase
            desc = [w.translate(table) for w in desc]          # remove punctuation
            desc = [w for w in desc if len(w) > 1]      # < 2 words in a sentence
            desc = [w for w in desc if w.isalpha()]     # only keep alphabets

            desc_list[i] = ' '.join(desc)

def save_descriptions(desc, dir, fname):
    lines = list()
    for id, desc_list in desc.items():
        for desc in desc_list:
            line = id + ' ' + desc
            lines.append(line)
    data = '\n'.join(lines)

    if not os.path.exists(dir):
        os.makedirs(dir)

    fname = os.path.join(dir, fname)
    with open(fname, 'w') as text_file:
        text_file.write(data)

if __name__ == '__main__':

    # Test 1
    inp_fname = 'Dataset/Flickr8k_text/Flickr8k.token.txt'
    out_dir = 'ImgDescriptions'
    out_fname = 'Flickr8k.token_tmp.txt'
    doc = load_doc(inp_fname)
    if doc:
        desc = load_descriptions(doc)
        clean_text(desc)
        save_descriptions(desc, out_dir, out_fname)
"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from tqdm import tqdm

word_file = open('vocab_words.txt','r')
vocab = []
for line in word_file.readlines():
    vocab.append(line.split(', ')[0])

index2word = {}
word2index = {}
for i,word in enumerate(vocab):
    index2word[i] = word
    word2index[word] = i

datas = json.load(open('data/annotations/dataset_coco.json','r'))
images = datas['images']
visual_concepts = []
for img in tqdm(images):
    sents = img['sentences']
    tokens = []
    tag = [0]*1000
    for sent in sents:
        tokens.extend(sent['tokens'])
    tokens = list(set(tokens))
    for token in tokens:
        if token in vocab:
            tag[word2index[token]] = 1
    #print(tag)
    visual_concepts.append(tag)

file = open('visual_concepts.json','w')
json.dump(visual_concepts,file)
print('finish')
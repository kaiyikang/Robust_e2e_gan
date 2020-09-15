#%%
import os
import numpy as np
# %%
word_dim = 0 
path = r"/home/kang/Develop/Robust_e2e_gan/k2k/data/sgns.wiki.char_2"
with open(path, 'r', encoding='utf-8') as fid:
    for line in fid:
        line_splits = line.strip().split()               
        word_dim = int(line_splits[1])
        break 
shape = (2000, word_dim)

scale = 0.05

embed_vecs_init = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
input_unit = word_dim

with open(path, 'r', encoding='utf-8') as fid:
    for line in fid:
        line_splits = line.strip().split()
        char = line_splits[0]
        print(char)
            
      
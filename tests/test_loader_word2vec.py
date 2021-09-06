import os
_path = os.getcwd()
import sys
sys.path.append(_path)
import numpy as np

from model.loader import word2vec

texts = ['今天晚上吃什么','明天晚上吃什么','今天晚上跑步吗','白天去不去图书馆','去酒吧度一晚上']
vec = np.array([])
for _ in (word2vec(text) for text in texts):
    print(_)
    vec = np.append(vec,_)
print(vec, len(vec), vec[0], vec[0].size)

vector = [word2vec(text) for text in texts]
print(type(vector[0]),vector[0].size,vector[0])
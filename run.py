import os
import numpy as np

from utils.keywords4doc import Processor
from model.loader import word2vec
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

path_kw_dict = os.getcwd()+'/data/keywords_dict.txt'
DOC = '咖啡中成分对于高血尿酸病患的作用分析'

doc = str(input('输入为： （输入''/''，则为内置默认输入''咖啡中成分对于高血尿酸病患的作用分析''）'))
round = int(input('搜索关键词字典的前几个？提示：速度约为45秒/100个'))

if doc=='/':
    doc = DOC
doc_kwords = Processor(doc).keywords_max_sum_sim(3,10)

with open(path_kw_dict) as f:
    kw_dict = eval(f.read())
    print('Keywords dictionary started')
    cnt = 0
    scores = np.array([])
    for title, kwords in kw_dict.items():
        if cnt==0:
            print('Search started')
        '''
        if kwords==[]:
            kwords = Processor(title).keyword_max_sum_sim(3,10)'''

        score_list = np.array([cos_sim(word2vec(i),word2vec(j)) for i in kwords for j in doc_kwords])
        score = np.sum(np.sort(score_list)[-5:])
        scores = np.append(scores,score)
        print('round: ',cnt+1,', get score: ',score)
        cnt += 1
        if cnt>round:
            break

    recommend_index = scores.argsort()[-5:]
    recommend_list = [list(kw_dict.keys())[idx] for idx in recommend_index]
    
    print('Task done, result=')
    print(recommend_list)
    
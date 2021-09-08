import os
_path = os.getcwd()
import sys
sys.path.append(_path)
import warnings
warnings.filterwarnings('ignore')

from utils.keywords4doc import Processor

doc = '慧源共享全国高校开放数据创新研究大赛'

p = Processor(doc)

print('原文为: ', doc)
print('采用关键词基线模型： ',p.keywords_basic(4))
print('惩罚项为最大相似度和： ',p.keywords_max_sum_sim(4,12))
print('惩罚项为最大边际相关度： ',p.keywords_max_marg_rel(4,0.2))
import os
_path = os.getcwd()
import sys
sys.path.append(_path)
import warnings
warnings.filterwarnings('ignore')

from utils.keywords4doc import Processor

text = '基于人工智能的实况足球游戏战术模拟及实际应用'
processor = Processor(text)

print('test for method: cut, ',processor.cut())
print('test for method: candidates, ',processor.candidates())
print('test for method: doc_embedding, ',type(processor.doc_embedding()),processor.doc_embedding().size)
print('test for method: candidates_embedding_list, ',type(processor.candidates_embedding_list()),type(processor.candidates_embedding_list()[0]),processor.candidates_embedding_list()[0].size)
print('test for method: keywords_basic, ',processor.keywords_basic(4))
print('test for method: keywords_max_sum_sim, ',processor.keywords_max_sum_sim(4,12))
print('test for method: keywords_max_marg_rel, ',processor.keywords_max_marg_rel(4,0.7))
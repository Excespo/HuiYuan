import os
_path = os.getcwd()
import sys
sys.path.append(_path)

from model.loader import word2vec

import jieba
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

import itertools

class Processor():
    '''Create a processor for word segmentation and keywords extraction
    '''

    def __init__(self,doc):
        self.doc = doc
        self.n_gram_range = (1,1)
        self.candidates = self.candidates()
        self.doc_embedding = self.doc_embedding()
        self.candidates_embedding_list = self.candidates_embedding_list()

    def cut(self):
        '''cut with jieba
        '''
        return ' '.join(jieba.lcut_for_search(self.doc))

    def candidates(self):
        '''Select candidates for keywords
        '''
        count = CountVectorizer(ngram_range=self.n_gram_range).fit([self.cut()])
        _candidates = count.get_feature_names()
        return _candidates
    
    def doc_embedding(self):
        '''Caluculate embedding matrix for doc
        '''
        return word2vec(self.doc)

    def candidates_embedding_list(self):
        '''Calculate embedding matrix for every candidate
        '''
        return [word2vec(candidate) for candidate in self.candidates]
    
    def keywords_basic(self,top_n:int):
        '''Determine keywords of doc simply by somparing cosine similarity sum
        Args:
            top_n: number for top n choice
        Returns:
            topn_list: keywords list with top_n elements
        '''
        distances = np.array([cos_sim(self.doc_embedding,candidate) for candidate in self.candidates_embedding_list]).squeeze()
        topn_index = distances.argsort()[::-1][:top_n]
        topn_list = [self.candidates[idx] for idx in topn_index]
        return topn_list
    
    def keywords_max_sum_sim(self,top_n,nr_candidates):
        '''Consider additionally the inter-keywords cosine similarity sum
        Args:
            top_n: number for second round choice
            nr_candidates: number for first round choice
        Returns:
            topn_list: keywords list with top_n elements
        '''
        distances = np.array([cos_sim(self.doc_embedding,candidate) for candidate in self.candidates_embedding_list]).squeeze()
        tmp_topn_index = distances.argsort()[::-1][:nr_candidates]
        tmp_topn_list = [self.candidates[idx] for idx in tmp_topn_index]
        tmp_candidates_embedding_list = [self.candidates_embedding_list[idx] for idx in tmp_topn_index]

        min_sim = np.inf
        topn_index_list = []

        for combination in itertools.combinations(range(len(tmp_topn_list)),top_n):
            sim = sum([cos_sim(tmp_candidates_embedding_list[i],tmp_candidates_embedding_list[j]) for i in combination for j in combination if i!=j])
            if sim<min_sim:
                topn_index_list.append(combination)
                min_sim = sim
        
        topn_index = topn_index_list[-1]
        topn_list = [self.candidates[idx] for idx in topn_index]
        return topn_list
    
    def keywords_max_marg_rel(self,top_n,diversity):
        '''Consider additionally the maximum marginal relevance
        Args:
            top_n: number for top n choice
            diversity: the hyperparameter to determine the influence of maximum marginal relevance
        Returns:
            topn_list: keywords list with top_n elements
        '''
        
        min_mr = -np.inf
        topn_index_list = []

        for combination in itertools.combinations(range(len(self.candidates)),top_n):
            rel_doc_candidates = np.sum(np.array([cos_sim(self.doc_embedding,candidate) for candidate in self.candidates_embedding_list]))
            max_rel_inter_candidates = max([cos_sim(self.candidates_embedding_list[i],self.candidates_embedding_list[j]) for i in combination for j in combination if i!=j])
            mr = (1-diversity)*rel_doc_candidates-diversity*max_rel_inter_candidates
            if mr>min_mr:
                topn_index_list.append(combination)
                min_mr = mr
        
        topn_index = topn_index_list[-1]
        topn_list = [self.candidates[idx] for idx in topn_index]
        return topn_list

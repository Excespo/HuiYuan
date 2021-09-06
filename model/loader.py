import warnings
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, logging

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

MODEL_PATH = 'hfl/chinese-bert-wwm-ext'

def loader(model_path:str):
    '''Load BERT model
    Args:
        model_path: c.f. huggingface.co
    Returns:
        model,
        tokenizer
    '''
    model = BertModel.from_pretrained(model_path).to('cuda')
    tokenizer = BertTokenizer.from_pretrained(model_path)

    return model,tokenizer

model, tokenizer = loader(MODEL_PATH)

def word2vec(word:str):
    '''Transform word to vector

    Args:
        word: input
    Returns:
        vec: np.ndarray of size (1,768)
    '''
    inputs = tokenizer(word,return_tensors='pt').to('cuda')
    pooler_out = model(**inputs).pooler_output
    vec = pooler_out.detach().cpu().numpy()

    return vec

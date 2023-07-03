import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
similarity_model = SentenceTransformer(config.SIMILARITY_MODEL_NAME).to('cuda:0')


def similarity(text1, text2):
    sentences = [text1, text2] 
    embeddings = similarity_model.encode(sentences)
    cos_similarity = pytorch_cos_sim(embeddings[0], embeddings[1])

    return round(cos_similarity.item(), 2)

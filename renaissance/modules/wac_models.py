from typing import Unpack
from sklearn.linear_model import SGDClassifier
import numpy as np

class WACModels():

    def __init__(self, 
                 vocab: list[str], 
                 embedding_size: int, 
                 position_size: int, 
                 save_directory: str,
                 **wac_kwargs: Unpack[dict]):
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.positon_size = position_size
        self.save_directory = save_directory

        # Exclusionary labels dictionary to stop image
        # features from being used again once they
        # were used for training 
        self.used_feature_ids = {}

        # Keep track of active features to be used
        # in the next training round
        self.current_wac_datasets = {}
        self.current_feature_ids = {}
        
        # Create WAC models for each word and add 
        # an empty dataset, empty current feature ID, and 
        # empty past feature ID for each word
        self.wac_models = {}
        for word in self.vocab:
            self.wac_models[word] = SGDClassifier(**wac_kwargs)
            self.used_feature_ids[word] = set()
            self.current_feature_ids[word] = set()
            self.current_wac_datasets[word] = {"pos": [], "neg": []}

    def add_word_sample(self, word: str, feature_id: tuple, embedding: np.ndarray, label: int):

        if word not in self.vocab:
            raise ValueError(f"Word {word} is not in the WAC model vocabulary")
        
        else:
            if feature_id in self.used_feature_ids[word]:
                return
        
        self.current_feature_ids[word].add(feature_id)
        if label == 1:
            self.current_wac_datasets[word]["pos"].append(embedding)
        elif label == 0:
            self.current_wac_datasets[word]["neg"].append(embedding)
        else:
            raise ValueError(f"Label {str(label)} is not 0 (negative) or 1 (positive)")
        
    def add_positive(self, word, feature_id, embedding):
        self.add_word_sample(word, feature_id, embedding, 1)

    def sample_negatives(self):
        pass
    

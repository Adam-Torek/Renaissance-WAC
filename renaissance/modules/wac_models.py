from typing import Unpack
from sklearn.linear_model import SGDClassifier
import numpy as np
import random
import multiprocessing as mp
import os
import json
import pickle
import re
import math

class WACModels():

    def __init__(self, 
                 save_directory: str,
                 vocab: list[str], 
                 special_vocab: list[str],
                 embedding_size: int, 
                 position_size: int, 
                 neg_to_pos: int=3,
                 num_cores: int=-1,
                 **wac_kwargs: Unpack[dict],) -> None:
        
        self.save_directory = save_directory
        self.vocab = []
        self.special_vocab = []

        for word in vocab:
            word = re.sub(r"/", "{slash}", word)
            self.vocab.append(word)
        
        for word in special_vocab:
            self.special_vocab.append(word)

        self.embedding_size = embedding_size
        self.position_size = position_size
        self.save_directory = save_directory
        self.neg_to_pos = neg_to_pos
        self.num_cores = num_cores

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
            self.wac_models[word] = SGDClassifier(loss="log_loss",**wac_kwargs)
            self.used_feature_ids[word] = set()
            self.current_feature_ids[word] = set()
            self.current_wac_datasets[word] = None

    def add_word_sample(self, 
                        word: str, 
                        feature_id: tuple, 
                        embedding: np.ndarray, 
                        label: int) -> None:

        # Raise error if word is not defined in vocab
        if word not in self.vocab:
            raise ValueError(f"Word {word} is not in the WAC model vocabulary")
        
        if word in self.special_vocab:
            return

        # Do not add the visual feature to the WAC dataset to train if it already exists
        else:
            if label == 1 and feature_id in self.used_feature_ids[word] or feature_id in self.current_feature_ids[word]:
                return
        
        # Create the temporary training dataset dict if it does not exist
        if self.current_wac_datasets[word] is None:
            self.current_wac_datasets[word] = {"pos": [], "neg": []}

        # Add visual feature and feature ID to current WAC database
        if label == 1:
            self.current_wac_datasets[word]["pos"].append((embedding, feature_id))
            self.current_feature_ids[word].add(feature_id)
        elif label == 0:
            self.current_wac_datasets[word]["neg"].append((embedding, feature_id))
        else:
            raise ValueError(f"Label {str(label)} is not 0 (negative) or 1 (positive)")
        
    def add_positive(self, word, feature_id, embedding) -> None:
        self.add_word_sample(word, feature_id, embedding, 1)

    def sample_negatives(self) -> None:
        feature_sample_space = {}

        # Get the possible sample feature space for each word that
        # has positive features added to it 
        for word, dataset in self.current_wac_datasets.items():
            if dataset is None:
                continue
            feature_sample_space[word] = dataset["pos"]

        # Get the list of words with current datasets to sample from
        possible_words = list(feature_sample_space.keys())

        # Do the negative sampling for each word with a defined dataset
        for word, dataset in feature_sample_space.items():
            # Get the number of negatives to sample for this word
            num_negatives = len(self.current_wac_datasets[word]["pos"]) * self.neg_to_pos
            negative_samples = []

            # Do negative sampling for this current word 
            while len(negative_samples) < num_negatives:
                # Choose a negative word to sample from and skip over the sampled word
                # if it is the current word 
                negative_word = random.choice(possible_words)
                if negative_word == word:
                    continue

                # Get the negative word samples for the current word
                negative_word_samples = self.current_wac_datasets[negative_word]["pos"]

                # Cut down the number of possible negatives if it is greater than the 
                # number of negative samples 
                if len(negative_samples) > num_negatives:
                    for _ in range(0, negative_word_samples):
                        negative_word_samples.pop()
                
                # Get a random number of negative samples to collect and add them to the negatives samples list
                num_negatives_to_sample = math.ceil(random.uniform(1, len(negative_word_samples)))
                current_samples = 0
                while current_samples < num_negatives_to_sample:
                    negative_sample = random.choice(negative_word_samples)

                    # Avoid sampling features from a negative word that are also in this word as well
                    negative_feature_id = negative_sample[1]
                    if negative_feature_id in self.current_feature_ids[word]:
                        continue
                    else:
                        negative_samples.append(negative_sample)
                        current_samples += 1
            
            # Add the negative samples to the current word dataset 
            for sample in negative_samples:
                embedding, feature_id = sample
                self.add_word_sample(word, feature_id, embedding, 0)

    def _train_single_model(self, word: str) -> tuple:

        # Get the model to train and the assembled dataset
        model_to_train = self.wac_models[word]
        wac_dataset = self.current_wac_datasets[word]

        wac_training_features = []
        wac_training_labels = []

        # Assemble the dataset and feature IDs
        for feature_key in ["pos","neg"]:
            for feature in wac_dataset[feature_key]:

                # Get the feature embedding and feature ID 
                # And add them to the training dataset
                feature_embedding, feature_id = feature
                wac_training_features.append(feature_embedding)

                label = 1 if feature_key == "pos" else 0
                wac_training_labels.append(label)
        
        # Create the training dataset and labels for training
        wac_training_features = np.array(wac_training_features)
        wac_training_labels = np.array(wac_training_labels)

        # Train the model and return it, the word, and training IDs
        trained_model = model_to_train.fit(wac_training_features, wac_training_labels)

        return (word, trained_model)
    
    def _train_models_single_thread(self, words_to_use: list[str]) -> list:
        trained_model_list = []
        for word in words_to_use:
            trained_model_tuple = self._train_single_model(word)
            trained_model_list.append(trained_model_tuple)

        return trained_model_list
        
    def train_models(self) -> None:

        # Get the number of cores to use for multiprocessing
        if self.num_cores == -1:
            num_cores = os.cpu_count()
        else:
            num_cores = self.num_cores

        # Get the WAC models to train
        words_to_use = [key for key, value in self.current_wac_datasets.items() if value is not None]
        trained_model_list = []

        # Attempt to train models using multi-processing 
        if self.num_cores == -1 or self.num_cores > 0:
            try:
                with mp.Pool(processes=num_cores) as pool:
                    trained_model_list = pool.map(self._train_single_model, words_to_use)

            # Train models in single-threaded mode if an error occurs in multiprocess mode
            except Exception as e:
                print(f"Unable to use multiprocessing due to the following error: {str(e)}. Running in single process mode.")
                trained_model_list = self._train_models_single_thread(words_to_use)
        else:
            trained_model_list = self._train_models_single_thread(words_to_use)

        # Save the trained wac models, clear out temporary datasets, and add feature IDs to 
        # use feature IDS list
        for word, trained_model in trained_model_list:
            self.used_feature_ids[word].update(self.current_feature_ids[word])
            self.current_feature_ids[word] = None
            
            self.current_wac_datasets[word] = None
            self.wac_models[word] = trained_model

    def get_distributions(self, word_features: np.ndarray) -> dict:

        # Get a list of distributions for all words in the provided 
        # words dict
        probability_dict = {}
        for word, model in self.wac_models.items():
            word = re.sub("{slash}", "/", word)
            try:
                word_prob = model.predict_proba(word_features)[:,1]
            except Exception as e:
                word_prob = np.zeros((word_features.shape[0],))
            
            probability_dict[word] = word_prob

        for word in self.special_vocab:
            probability_dict[word] = np.zeros((word_features.shape[0],))
        
        return probability_dict

    def get_embeddings(self, words: list) -> np.array:

        # Get embeddings for each word from the WAC model if it exists, and a vector of zeros if it does not
        embeddings_list = []
        for word in words:
            word = re.sub(r"/", "{slash}", word)
            try:
                random_features = np.random((self.embedding_size + self.position_size + 1,))
                _ = self.wac_models[word].predict(random_features)
                word_embedding = np.concat([self.wac_datasets[word].coef_, self.wac_datasets[word].intercept_]) 
            except Exception as e:
                word_embedding = np.zeros((self.embedding_size+self.position_size+1,))

            embeddings_list.append(word_embedding)
        
        embeddings_list = np.array(embeddings_list)
        return embeddings_list

    def save_models(self) -> None:
        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)
        
        # Save WAC model metadata to disk
        wac_metadata = {}
        wac_metadata["embedding_size"] = self.embedding_size
        wac_metadata["position_size"] = self.position_size
        wac_metadata["vocab"] = self.vocab
        wac_metadata["special_vocab"] = self.special_vocab

        with open(os.path.join(self.save_directory, "wac_metadata.json"), "w") as json_file:
            json.dump(wac_metadata, json_file)

        # Save every WAC model to disk
        for word, model in self.wac_models.items():
            with open(os.path.join(self.save_directory, f"{word}.pkl"), "wb") as model_file:
                pickle.dump(model, model_file)

    def load_models(self) -> None:
        
        if not os.path.exists(self.save_directory):
            raise ValueError(f"Directory {self.save_directory} cannot be found or does not exist.")

        # Load WAC model metadata from disk
        with open(os.path.join(self.save_directory, "wac_metadata.json"), "r") as json_file:
            wac_metadata = json.load(json_file)
            self.embedding_size = wac_metadata["embedding_size"]
            self.position_size = wac_metadata["position_size"]
            self.vocab = wac_metadata["vocab"]
            self.special_vocab = wac_metadata["special_vocab"]

        # Load each WAC model from disk
        for word in self.vocab:
            with open(os.path.join(self.save_directory, f"{word}.pkl"), "rb") as model_file:
                self.wac_models[word] = pickle.load(model_file)
    
from typing import Unpack
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from multiprocessing import get_context
import os
import json
import pickle
import re
from mpi4py.futures import MPIPoolExecutor

class WACModels():

    def __init__(self, 
                 save_directory: str,
                 vocab: list[str], 
                 special_vocab: list[str],
                 embedding_size: int, 
                 position_size: int, 
                 neg_to_pos: int=3,
                 num_cores: int=-1,
                 save_wac_features=False,
                 wac_feature_splits: list[str] = ["train","val","test"],
                 **wac_kwargs: Unpack[dict],) -> None:
        
        self.save_directory = save_directory
        self.vocab = []
        self.special_vocab = []
        self.splits_to_use = wac_feature_splits
        self.current_split = "train"
        self.save_wac_features = save_wac_features

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

        # Exclusionary labels dictionary to stop positive WAC
        # features from being used for the given word 
        # multiple times
        self.positive_feature_ids = {}

        # Keep track of active features to be used
        # in the next training round
        self.current_wac_datasets = {}

        # Dictionary used to store WAC features
        self.wac_features = {}
        for split in self.splits_to_use:
            self.wac_features[split] = {}
            self.positive_feature_ids[split] = {} 
            self.current_wac_datasets[split] = {}
            for word in self.vocab:
                self.positive_feature_ids[split][word] = set()
                self.current_wac_datasets[split][word] = None

        self.scaler = MinMaxScaler()
        
        # Create WAC models for each word and add 
        # an empty dataset, empty current feature ID, and 
        # empty past feature ID for each word
        self.wac_models = {}
        for word in self.vocab:
            self.wac_models[word] = SGDClassifier(loss="log_loss", **wac_kwargs)

    def set_current_split(self, split: str) -> None:
        if split not in self.splits_to_use:
            raise ValueError(f"Split {split} is not an available split. You must select from the following: {str(self.splits_to_use)}")
        self.current_split = split

    def check_features_loaded(self):
        for split in self.splits_to_use:
            if len(self.wac_features[split]) == 0 or len(self.positive_feature_ids[split]) == 0:
                return False 
        
        return True

    def add_word_sample(self, 
                        word: str, 
                        feature_id: tuple, 
                        label: int) -> None:

        # Raise error if word is not defined in vocab
        if word not in self.vocab:
            raise ValueError(f"Word {word} is not in the WAC model vocabulary")
        
        if word in self.special_vocab:
            return

        # Do not add the visual feature to the WAC dataset to train if it already exists
        else:
            if label == 1 and feature_id in self.positive_feature_ids[self.current_split][word]:
                return
        
        # Create the temporary training dataset dict if it does not exist
        if self.current_wac_datasets[self.current_split][word] is None:
            self.current_wac_datasets[self.current_split][word] = {"pos": [], "neg": []}

        # Add visual feature and feature ID to current WAC database
        if label == 1:
            self.current_wac_datasets[self.current_split][word]["pos"].append(feature_id)

        elif label == 0:
            self.current_wac_datasets[self.current_split][word]["neg"].append(feature_id)
        else:
            raise ValueError(f"Label {str(label)} is not 0 (negative) or 1 (positive)")
        
    def add_features(self, 
                     feature_ids: list[int], 
                     wac_features: list[np.ndarray],
                     tokenized_words: list[list[str]],
                     split: str) -> None:
        
        for feature_id, wac_feature in zip(feature_ids, wac_features):
            self.wac_features[split][feature_id] = wac_feature

        for sentence in tokenized_words:
            for word in sentence:
                for feature_id in feature_ids:
                    if word not in self.positive_feature_ids[split]:
                        self.positive_feature_ids[split][word] = set()

                    self.positive_feature_ids[split][word].add(feature_id)
        
    def add_positive(self, word: str, feature_id: int) -> None:
        self.add_word_sample(word, feature_id, 1)

    def _sample_word_negatives(self, word: str) -> tuple:
        num_negatives = len(self.current_wac_datasets[self.current_split][word]["pos"]) * self.neg_to_pos
        negative_samples = []

        # Get a possible list of negative features to sample
        sample_space = list(self.wac_features[self.current_split].keys())
        for feature_id in self.positive_feature_ids[self.current_split][word]:
            if feature_id in sample_space:
                sample_space.remove(feature_id)

        # Get the actual sample of negative features for this word
        negative_samples = random.sample(sample_space, k=num_negatives)
        return (word, negative_samples)
    
    def _update_wac_model_word(self, word: str, positive_feature_ids: list[int]) -> None:
        positive_features = []
        for pos_feature_id in positive_feature_ids:
            self.positive_feature_ids[self.current_split][word].add(pos_feature_id)
            positive_features.append(pos_feature_id)
        
        num_negative_features = len(positive_features) * self.neg_to_pos
        negative_feature_space = set(self.wac_features[self.current_split].keys())

        for pos_feature_id in self.positive_feature_ids[self.current_split][word]:
            if pos_feature_id in negative_feature_space:
                negative_feature_space.remove(pos_feature_id)

        negative_feature_space = list(negative_feature_space)

        negative_feature_ids = list(random.sample(negative_feature_space, k=num_negative_features))

        feature_dataset = []
        feature_labels = []

        for pos_feature_id in positive_feature_ids:
            feature_dataset.append(self.wac_features[self.current_split][pos_feature_id])
            feature_labels.append(1)
        
        for neg_feature_id in negative_feature_ids:
            feature_dataset.append(self.wac_features[self.current_split][neg_feature_id])
            feature_labels.append(0)

        feature_dataset = np.array(feature_dataset)
        feature_labels = np.expand_dims(np.array(feature_labels), axis=1)

        complete_dataset = np.concatenate([feature_dataset, feature_labels], axis=1)
        randomizer = np.random.default_rng()
        randomizer.shuffle(complete_dataset, axis=0)

        randomized_features = complete_dataset[:, :-1]
        randomized_labels =complete_dataset[:, -1]

        randomized_features = self.scaler.fit_transform(randomized_features)
        model_to_train = self.wac_models[word]
        model_to_train.fit(randomized_features, randomized_labels)

        return (word, model_to_train, positive_feature_ids)

    def update_wac_models(self, word_feature_ids: dict) -> None:
        num_cores = os.cpu_count() if self.num_cores == -1 else self.num_cores
        trained_model_results = []
        if num_cores == 0:
            for word, pos_feature_ids in word_feature_ids.items():
                word_model = self._update_wac_model_word(word, pos_feature_ids)
                trained_model_results.append(word_model)
        else:
            try:
                with get_context("spawn").Pool(processes=num_cores) as process_pool:
                    training_model_arguments = list(word_feature_ids.items())
                    trained_model_results = process_pool.starmap(self._update_wac_model_word, training_model_arguments)
            except Exception as e:
                print(f"Unable to use multi-process approach due to the following exception: {str(e)}. \
                        Using single-process approach instead.")
                for word, pos_feature_ids in word_feature_ids.items():
                    word_model = self._update_wac_model_word(word, pos_feature_ids)
                    trained_model_results.append(word_model)
        
        for word, model, pos_feature_ids in trained_model_results:
            self.wac_models[word] = model
            self.positive_feature_ids[self.current_split][word].update(pos_feature_ids)

    def sample_negatives(self) -> None:
        feature_sample_space = []

        # Get the possible sample feature space for each word that
        # has positive features added to it 
        for word, dataset in self.current_wac_datasets[self.current_split].items():
            if dataset is None or len(dataset["pos"]) == 0:
                continue
            feature_sample_space.append(word)

        negative_word_samples = []
        # Do the negative sampling for each word with a defined dataset
        if self.num_cores == -1 or self.num_cores > 0:
            num_cores = os.cpu_count() if self.num_cores == -1 else self.num_cores
            try: 
                with get_context("spawn").Pool(processes=num_cores) as process_pool:
                    negative_word_samples = process_pool.map(self._sample_word_negatives, feature_sample_space)
            except Exception as e:
                print(f"Unable to do multi-process negative word sampling due to the following error: {str(e)}. \
                        Performing word sampling in single-process mode instead.")
                for word in feature_sample_space:
                    negative_word_samples.append(self._sample_word_negatives(word))
        else:
            for word in feature_sample_space:
                negative_word_samples.append(self._sample_word_negatives(word))

        for word, negative_samples in negative_word_samples:
            for feature_id in negative_samples:
                self.add_word_sample(word, feature_id, 0)

    def _train_single_model(self, 
                            word: str, 
                            wac_training_features: np.array, 
                            wac_training_labels: np.array) -> tuple:

        # Train the model and return it, the word, and training IDs
        model_to_train = self.wac_models[word]
        wac_training_features = self.scaler.fit_transform(wac_training_features)
        model_to_train.fit(wac_training_features, wac_training_labels)

        return (word, model_to_train)
    
    def _train_models_single_thread(self, datasets_to_train: list[tuple]) -> list:
        trained_model_list = []
        for word, training_features, training_labels in datasets_to_train:
            trained_model_tuple = self._train_single_model(word, training_features, training_labels)
            trained_model_list.append(trained_model_tuple)

        return trained_model_list
        
    def train_models(self) -> None:

        # Get the WAC models to train
        datasets_to_train = []
        for word, dataset in self.current_wac_datasets[self.current_split].items():
            if dataset is None:
                continue
            wac_training_features = []
            wac_training_labels = []
            for feature_key in ["pos","neg"]:
                for feature_id in dataset[feature_key]:
                    feature_embedding = self.wac_features[self.current_split][feature_id]
                    wac_training_features.append(feature_embedding)

                    label = 1 if feature_key == "pos" else 0
                    wac_training_labels.append(label)
            
            wac_training_features = np.array(wac_training_features)
            wac_training_labels = np.array(wac_training_labels)
            datasets_to_train.append((word, wac_training_features, wac_training_labels))
    
        # Attempt to train models using multi-processing 
        if self.num_cores == -1 or self.num_cores > 0:
            try:
                num_cores = os.cpu_count() if self.num_cores == -1 else self.num_cores
                with get_context("spawn").Pool(processes=num_cores) as process_pool:
                    trained_model_list = process_pool.starmap(self._train_single_model, datasets_to_train)
            # Train models in single-threaded mode if an error occurs in multiprocess mode
            except Exception as e:
                print(f"Unable to use multiprocessing due to the following error: {str(e)}. Running in single process mode.")
                trained_model_list = self._train_models_single_thread(datasets_to_train)
        else:
            trained_model_list = self._train_models_single_thread(datasets_to_train)

        # Save the trained wac models, clear out temporary datasets, and add feature IDs to 
        # use feature IDS list
        for word, trained_model in trained_model_list:
            self.current_wac_datasets[self.current_split][word] = None
            self.wac_models[word] = trained_model

    def get_distributions(self, indices: list[int]) -> dict:

        # Get a list of distributions for all words in the provided 
        # words dict
        word_features = []
        for index in indices:
            word_features.append(self.wac_features[self.current_split][index])
        word_features = np.stack(word_features)

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
                generator = np.random.default_rng()
                random_features = generator.random((1, self.embedding_size + self.position_size,))
                _ = self.wac_models[word].predict(random_features)
                word_embedding = np.concat([np.squeeze(self.wac_models[word].coef_), self.wac_models[word].intercept_]) 
            except Exception as e:
                word_embedding = np.full((self.embedding_size+self.position_size+1,), 1e-10)

            embeddings_list.append(word_embedding)
        
        embeddings_list = np.stack(embeddings_list)
        return embeddings_list

    def save_models(self) -> None:
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory, exist_ok=True)
        
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

    def load_models(self, load_directory=None) -> None:
        
        if load_directory is None or not os.path.exists(load_directory):
            directory_to_use = self.save_directory
        else:
            directory_to_use = load_directory

        if not os.path.exists(directory_to_use):
            raise ValueError(f"Directory {directory_to_use} cannot be found or does not exist.")

        # Load WAC model metadata from disk
        with open(os.path.join(directory_to_use, "wac_metadata.json"), "r") as json_file:
            wac_metadata = json.load(json_file)
            self.embedding_size = wac_metadata["embedding_size"]
            self.position_size = wac_metadata["position_size"]
            self.vocab = wac_metadata["vocab"]
            self.special_vocab = wac_metadata["special_vocab"]

        # Load each WAC model from disk
        for word in self.vocab:
            with open(os.path.join(directory_to_use, f"{word}.pkl"), "rb") as model_file:
                self.wac_models[word] = pickle.load(model_file)

    def save_features(self):
         if self.save_wac_features:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory, exist_ok=True)
            
            with open(os.path.join(self.save_directory, "wac_features.pkl"), "wb") as features_file:
                pickle.dump(self.wac_features, features_file)

            with open(os.path.join(self.save_directory, "positive_feature_ids.pkl"), "wb") as feature_ids_file:
                pickle.dump(self.positive_feature_ids, feature_ids_file)

    def load_features(self):
        if self.save_wac_features:
            try:
                with open(os.path.join(self.save_directory, "wac_features.pkl"), "rb") as features_file:
                    self.wac_features = pickle.load(features_file)

                with open(os.path.join(self.save_directory, "positive_feature_ids.pkl"), "rb") as feature_ids_file:
                    self.positive_feature_ids = pickle.load(feature_ids_file)

                return True

            except Exception as e:
                return False
        
        else:
            return False
        

    
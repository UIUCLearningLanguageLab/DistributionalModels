import copy
import pickle
import pandas as pd
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset


class Corpus(Dataset):

    def __init__(self, document_headers=None):
        self.document_headers = document_headers

        self.num_documents = 0  # 3
        self.document_info_df = None  # index, name, types, tokens
        self.document_list = None  # list of sequence lists, which are lists of tokens

        self.num_sequences = None  # 6
        self.num_types = None  # number of unique tokens in entire corpus
        self.num_tokens = None
        self.corpus_freq_dict = None

        self.vocab_list = None
        self.vocab_index_dict = None
        self.vocab_size = None  # user specified number of types you want in the vocab
        self.unknown_token = None  # <unk>

        self.init_corpus()

        self.x_list = None  # the 1D list of indexes
        self.y_list = None

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.x_list[idx], dtype=torch.long)
        output_data = torch.tensor(self.y_list[idx], dtype=torch.long)
        return input_data, output_data

    def init_corpus(self):
        self.document_list = []
        header_list = ["name", "num_sequences", "num_types", "num_tokens"]
        if self.document_headers is not None:
            header_list += self.document_headers
        self.document_info_df = pd.DataFrame(columns=header_list)
        self.num_sequences = 0
        self.num_types = 0
        self.num_tokens = 0
        self.corpus_freq_dict = Counter()

    @staticmethod
    def get_document_string_from_file(document_path):
        with open(document_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def add_document(self, sequence_list, tokenized=False, document_info_dict=None):

        if document_info_dict is None:
            document_info_dict = {}

        # seqeunce_list = [[a11, y, b12, .], [a11, y, b12, .]]

        document_freq_dict = Counter()
        num_tokens = 0
        sequence_token_list = []
        for sequence in sequence_list:
            if tokenized:
                token_list = sequence
            else:
                token_list = self.tokenize(sequence)

            document_freq_dict.update(token_list)
            sequence_token_list.append(token_list)
            num_tokens += len(token_list)

        self.document_list.append(sequence_token_list)
        self.corpus_freq_dict.update(document_freq_dict)

        if 'name' not in document_info_dict:
            document_info_dict['name'] = len(document_info_dict)

        num_sequences = len(sequence_list)
        num_types = len(document_freq_dict)
        document_info_dict['num_sequences'] = num_sequences
        document_info_dict['num_types'] = num_types
        document_info_dict['num_tokens'] = num_tokens

        new_doc_df = pd.DataFrame([document_info_dict])

        self.document_info_df = pd.concat([self.document_info_df, new_doc_df], ignore_index=True)
        self.num_sequences += num_sequences
        self.num_tokens += num_tokens
        self.num_types = len(self.corpus_freq_dict)
        self.num_documents += 1

    def set_unknown_token(self, unknown_token="<unk>"):
        while self.unknown_token is None:
            if unknown_token in self.corpus_freq_dict:
                unknown_token = "<" + unknown_token + ">"
            else:
                self.unknown_token = unknown_token

    def create_vocab(self, vocab_size=None, include_list=(), exclude_list=(), include_unknown=True):
        print(f"Creating vocab list of size {vocab_size} and include_unknown={include_unknown}")

        # create the empty vocab list structures
        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_size = 0
        missing_word_list = []

        # if vocab_size is None, set it to the size of the freq_dict so all words are used
        # account for the unknown token if it will be included
        if vocab_size is None:
            if include_unknown:
                vocab_size = len(self.corpus_freq_dict) + 1
            else:
                vocab_size = len(self.corpus_freq_dict)

        # add unknown token to vocab
        if include_unknown:
            self.set_unknown_token()
            self.add_token_to_vocab(self.unknown_token)

        # get a filtered copy of the freq_dict that does not include any excluded words
        filtered_freq_dict = copy.deepcopy(self.corpus_freq_dict)
        for token in exclude_list:
            filtered_freq_dict.pop(token, None)
        if len(filtered_freq_dict) == 0:
            raise ValueError("ERROR making vocab list: After exclusion list there are no words in the corpus")

        # add words from the include list to the vocab data structures as long as they are in the filtered freq_dict
        for token in include_list:
            if token in filtered_freq_dict:
                self.add_token_to_vocab(token)
                filtered_freq_dict.pop(token, None)
            else:
                missing_word_list.append(token)

        # Add items from the counter to vocab_list it is not vocab_size
        if vocab_size > self.vocab_size:

            # Sort the counter by frequency (count), then by word
            sorted_tokens = sorted(filtered_freq_dict, key=lambda new_word: (-filtered_freq_dict[new_word], new_word))

            # Add words to vocab_list in frequency order until it reaches size m
            for token in sorted_tokens:
                if self.vocab_size >= vocab_size:
                    break
                if token not in self.vocab_index_dict:
                    self.add_token_to_vocab(token)

        return missing_word_list

    def add_token_to_vocab(self, token):
        self.vocab_list.append(token)
        self.vocab_index_dict[token] = self.vocab_size
        self.vocab_size += 1

    def flatten_corpus_lists(self, nested_list):
        # take an embedded list of whatever depth of embedding, and flatten into a single list
        flat_list = []
        for element in nested_list:
            if isinstance(element, list):
                # If the element is a list, extend flat_list with the flattened element
                flat_list.extend(self.flatten_corpus_lists(element))
            else:
                # If the element is not a list, add it to flat_list
                flat_list.append(element)
        return flat_list

    @staticmethod
    def create_simple_sequence_list(flattened_list, vocab_index_dict, unknown_token):
        sequence_list = []
        for token in flattened_list:
            if token in vocab_index_dict:
                current_index = vocab_index_dict[token]
            else:
                current_index = vocab_index_dict[unknown_token]

            sequence_list.append(current_index)
        return sequence_list

    @staticmethod
    def create_windowed_sequence_list(sequence_list, window_size):
        if window_size == 0:
            raise ValueError("Window size cannot be 0, must be None or positive integer")
        x = []
        y = []
        for i in range(len(sequence_list)):
            for j in range(1, window_size + 1):
                x.append(sequence_list[i])
                # Check if the index is within the bounds of the list
                if i - j >= 0:
                    y.append(sequence_list[i - j])
                if i + j < len(sequence_list):
                    y.append(sequence_list[i + j])
        return x, y

    @staticmethod
    def create_padded_sequence_list(sequence_list, sequence_length, pad_index):
        # Add padding at the beginning and end of the sequence list
        padded_sequence_list = [pad_index] * sequence_length + sequence_list + [pad_index] * sequence_length

        # Create sequences for x
        x = []
        y = []
        for i in range(len(padded_sequence_list) - sequence_length):
            x_seq = padded_sequence_list[i:i + sequence_length]
            x.append(x_seq)
            y_seq = padded_sequence_list[i + sequence_length]
            y.append([y_seq])
        return x, y


    def create_sequence_list(self, flattened_list, vocab_index_dict, unknown_token, window_size=None):
        sequence_list = self.create_simple_sequence_list(flattened_list, vocab_index_dict, unknown_token)
        if window_size is not None:
            x, y = self.create_windowed_sequence_list(sequence_list)
        else:
            x = sequence_list[:-1]
            y = sequence_list[1:]
        return x, y

    @staticmethod
    def tokenize(text_string):
        token_list = text_string.split()
        return token_list

    def save_to_pkl_file(self, file_path):
        print(f"Saving corpus to pkl {file_path}")
        """Save the instance to a file."""
        with open(file_path+'.pkl', 'wb') as file:
            pickle.dump(self, file)

    def save_to_txt_file(self, file_path):
        print(f"Saving corpus to txt {file_path}")
        """Save the instance to a file."""
        with open(file_path+'.txt', 'w') as file:
            for document in self.document_list:
                flattened_list = self.flatten_corpus_lists(document)
                document_string = " ".join(flattened_list)
                file.write(document_string + "\n")

    def save_to_csv_file(self, file_path):
        print(f"Saving corpus to csv {file_path}")
        """Save the instance to a file."""
        tuples_list = [(row.Index, row.name, row.age) for row in self.document_info_df.itertuples()]

        with open(file_path+'.csv', 'w') as file:
            for current_tuple in tuples_list:

                document = self.document_list[current_tuple[0]]
                doc_name = current_tuple[1]
                age = current_tuple[2]
                flattened_list = self.flatten_corpus_lists(document)
                document_string = " ".join(flattened_list)
                output_string = f"{doc_name},{age},{document_string}\n"
                file.write(output_string)

    @classmethod
    def load_from_file(cls, file_path):
        print(f"Loading corpus from {file_path}")
        """Load the instance from a file."""
        with open(file_path, 'rb') as file:
            return pickle.load(file)

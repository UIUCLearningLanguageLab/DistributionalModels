import copy
import pickle
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset
import sys

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

        self.index_list = None
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
    def create_simple_index_list(flattened_list, vocab_index_dict, unknown_token):
        index_list = []
        for token in flattened_list:
            if token in vocab_index_dict:
                current_index = vocab_index_dict[token]
            else:
                current_index = vocab_index_dict[unknown_token]

            index_list.append(current_index)
        return index_list

    @staticmethod
    def create_windowed_index_list(index_list, window_size):
        if window_size == 0:
            raise ValueError("Window size cannot be 0, must be None or positive integer")
        x = []
        y = []
        for i in range(len(index_list)):
            for j in range(1, window_size + 1):
                # Check if the index is within the bounds of the list
                if i - j >= 0:
                    x.append(index_list[i])
                    y.append(index_list[i - j])
                if i + j < len(index_list):
                    x.append(index_list[i])
                    y.append(index_list[i + j])
        return x, y

    def create_index_list(self, flattened_list, vocab_index_dict, unknown_token, window_size=None):
        index_list = self.create_simple_index_list(flattened_list, vocab_index_dict, unknown_token)
        if window_size is not None:
            x, y = self.create_windowed_index_list(index_list, window_size)
        else:
            x = index_list[:-1]
            y = index_list[1:]
        return x, y, index_list

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

    @staticmethod
    def create_sequence_lists(index_list, sequence_length, pad_index):
        if sequence_length == 2:
            # Each sequence is a single element from the index_list
            return [[index] for index in index_list]
        else:
            # Original logic for longer sequences
            padded_list = [pad_index] * (sequence_length - 2) + index_list + [pad_index] * (sequence_length - 2)
            sequence_lists = []
            for i in range(len(index_list) + 1):
                sequence = padded_list[i:i + sequence_length]
                sequence_lists.append(sequence)
            return sequence_lists

    @staticmethod
    def create_batches(sequence_list, batch_size, sequence_length, pad_index):
        x_batches = []
        y_batches = []
        y_window_batches = []
        current_batch_x = []
        current_batch_y = []
        current_batch_y_window = []

        if sequence_length == 1:
            for i in range(len(sequence_list) - 1):
                current_batch_x.append(sequence_list[i])
                current_batch_y.append(sequence_list[i + 1])
                current_batch_y_window.append(sequence_list[i + 1])

                if len(current_batch_x) == batch_size:
                    x_batches.append(current_batch_x)
                    y_batches.append(current_batch_y)
                    y_window_batches.append(current_batch_y)
                    current_batch_x = []
                    current_batch_y = []
                    current_batch_y_window = []
        else:
            for sequence in sequence_list:
                current_batch_x.append(sequence[:-1])  # Take all but the last element
                current_batch_y.append([sequence[-1]])   # Take the last element
                current_batch_y_window.append(sequence[1:])
                if len(current_batch_x) == batch_size:
                    x_batches.append(current_batch_x)
                    y_batches.append(current_batch_y)
                    y_window_batches.append(current_batch_y_window)
                    current_batch_x = []
                    current_batch_y = []
                    current_batch_y_window = []

            # Pad the last batch if necessary. this last bit is missing the completion for y_window_batches
            if current_batch_x:
                while len(current_batch_x) < batch_size:
                    current_batch_x.append([pad_index] * sequence_length)
                    current_batch_y.append([pad_index])
                    current_batch_y_window.append([pad_index] * sequence_length)

                x_batches.append(current_batch_x)
                y_batches.append(current_batch_y)
                y_window_batches.append(current_batch_y_window)

        return x_batches, y_batches, y_window_batches

    def create_batched_sequence_lists(self, document_list, window_size, batch_size, sequence_length, device):
        corpus_token_list = self.flatten_corpus_lists(document_list)
        pad_index = 0

        self.x_list, self.y_list, self.index_list = self.create_index_list(corpus_token_list,
                                                                           self.vocab_index_dict,
                                                                           self.unknown_token,
                                                                           window_size=window_size)
        if window_size == 1:
            sequence_list = self.create_sequence_lists(self.index_list, sequence_length+1, pad_index=pad_index)

            # print("sequence_list", sequence_list)

            x_batches, y_batches, y_window_batches = self.create_batches(sequence_list, batch_size, sequence_length,
                                                                         pad_index)
        else:
            x_batches = [[self.x_list[i:i + batch_size]] for i in range(0, len(self.x_list), batch_size)]
            y_batches = [[self.y_list[i:i + batch_size]] for i in range(0, len(self.y_list), batch_size)]
            y_window_batches = []

        #
        # print()
        # print("x_batches", x_batches)
        # print()
        # print("y_batches", y_batches)
        # print()
        # print("y_window_batches", y_window_batches)
        # sys.exit()

        x_batches = [torch.tensor(x_batch, dtype=torch.long).to(device) for x_batch in x_batches]
        y_batches = [torch.tensor(y_batch, dtype=torch.long).to(device) for y_batch in y_batches]
        y_window_batches = [torch.tensor(y_window_batch, dtype=torch.long).to(device) for y_window_batch in
                            y_window_batches]

        return x_batches, y_batches, y_window_batches

from torch.utils.data import Dataset
import torch
from ..corpora import corpus2

class CustomDataset(Dataset):

    # the x's could be:
    #   a token list that will be converted into
    #       an index list
    #       a list of one hot vectors
    #   a list of distributed vectors

    # the x's could be:
    #   a token list that will be converted into
    #       an index list
    #       a list of one hot vectors
    #   a list of distributed vectors

    def __init__(self):
        self.index_list = None

        self.x_list = None
        self.y_list = None
        self.x_index_dict = None
        self.y_index_dict = None
        self.num_x = None
        self.num_y = None
        self.unknown_token = None

        self.x_sequence_list = None  # the 1D list of indexes
        self.y_sequence_list = None

    def __len__(self):
        return len(self.x_sequence_list)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.x_sequence_list[idx], dtype=torch.long)
        output_data = torch.tensor(self.y_sequence_list[idx], dtype=torch.long)
        return input_data, output_data

    def create_types_from_corpus(self, corpus, variable, unknown_token=None):

        if corpus.vocab_list is not None:
            if corpus.unknown_token is not None:
                self.unknown_token = corpus.unknown_token
                token_list = corpus.vocab_list
            else:
                self.unknown_token = unknown_token
                token_list = [self.unknown_token] + corpus.vocab_list
        else:
            self.unknown_token = unknown_token
            if self.unknown_token is None:
                token_list = list(corpus.freq_dict.keys())
            else:
                token_list = [self.unknown_token] + list(corpus.freq_dict.keys())

        if variable == "x":
            self.x_list = token_list
            self.x_index_dict = {element: index for index, element in enumerate(token_list)}
            self.num_x = len(self.x_list)
        elif variable == "y":
            self.y_list = token_list
            self.y_index_dict = {element: index for index, element in enumerate(token_list)}
            self.num_y = len(self.y_list)


    def create_simple_index_list(self, flattened_list):
        index_list = []
        for token in flattened_list:
            if token in self.x_index_dict:
                current_index = self.x_index_dict[token]
            else:
                current_index = self.x_index_dict[self.unknown_token]

            index_list.append(current_index)
        return index_list

    @staticmethod
    def create_sequence_lists(index_list, sequence_length, pad_index):
        if sequence_length == 2:
            # Each sequence is a single element from the index_list
            return [[index] for index in index_list]
        else:
            # Original logic for longer sequences
            padded_list = [pad_index] * (sequence_length - 2) + index_list
            sequence_lists = []
            for i in range(len(padded_list) + 1):
                if i + sequence_length <= len(padded_list):
                    sequence = padded_list[i:i + sequence_length]
                    sequence_lists.append(sequence)
            return sequence_lists

    @staticmethod
    def create_windowed_index_list(index_list, window_size, direction='both', pad_index=0):
        if window_size == 0:
            raise ValueError("Window size cannot be 0, must be None or positive integer")
        # if direction == 'both':
        #     padded_index_list = [pad_index] * (window_size/2) + index_list
        if direction == 'backward':
            padded_index_list = [pad_index] * window_size + index_list
        else:
            padded_index_list = index_list + [pad_index] * window_size
        x = []
        y = []
        for i in range(len(padded_index_list)):
            for j in range(1, window_size + 1):
                # Check if the index is within the bounds of the list
                if direction == 'both':
                    if i - j >= 0:
                        x.append(padded_index_list[i])
                        y.append(padded_index_list[i - j])
                    if i + j < len(padded_index_list):
                        x.append(padded_index_list[i])
                        y.append(padded_index_list[i + j])
                elif direction == 'forward':
                    if i + window_size < len(padded_index_list):
                        x.append(padded_index_list[i])
                        y.append(padded_index_list[i + j])
                else:
                    if i < len(padded_index_list) - window_size:
                        x.append(padded_index_list[i + j - 1])
                        y.append(padded_index_list[i + window_size])
        return x, y

    def create_index_list(self, flattened_list, window_size=None, window_direction=None):
        index_list = self.create_simple_index_list(flattened_list)
        if window_size is not None:
            x, y = self.create_windowed_index_list(index_list, window_size, window_direction)
        else:
            x = index_list[:-1]
            y = index_list[1:]
        return x, y, index_list

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
                current_batch_y.append([sequence[-1]])  # Take the last element
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

    def create_batched_sequence_lists(self, document_list, window_size, window_direction, batch_size, sequence_length,
                                      device):
        corpus_token_list = corpus2.Corpus.flatten_corpus_lists(document_list)
        pad_index = 0
        window_size = window_size
        window_direction = window_direction
        self.x_sequence_list, self.y_sequence_list, self.index_list = self.create_index_list(corpus_token_list,
                                                                           window_size=window_size,
                                                                           window_direction=window_direction)
        if window_size == 1:
            sequence_list = self.create_sequence_lists(self.index_list, sequence_length + 1, pad_index=pad_index)

            x_batches, y_batches, y_window_batches = self.create_batches(sequence_list, batch_size, sequence_length,
                                                                         pad_index)
        else:
            x_batches = [[self.x_sequence_list[i:i + batch_size]] for i in range(0, len(self.x_sequence_list), batch_size)]
            y_batches = [[self.y_sequence_list[i:i + batch_size]] for i in range(0, len(self.y_sequence_list), batch_size)]
            y_window_batches = []

        x_batches = [torch.tensor(x_batch, dtype=torch.long).to(device) for x_batch in x_batches]
        y_batches = [torch.tensor(y_batch, dtype=torch.long).to(device) for y_batch in y_batches]
        y_window_batches = [torch.tensor(y_window_batch, dtype=torch.long).to(device) for y_window_batch in
                            y_window_batches]

        return x_batches, y_batches, y_window_batches
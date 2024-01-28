import os
import gzip
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_list):

        super(NeuralNetwork, self).__init__()
        self.vocab_list = vocab_list
        self.layer_dict = torch.nn.ModuleDict()
        self.state_dict = None
        self.device = None
        self.criterion = None
        self.optimizer = None
        self.model_name = None

        self.vocab_list = None
        self.vocab_index_dict = None
        self.vocab_size = None

        self.epoch = 0

        self.init_vocab(vocab_list)

    def init_vocab(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.vocab_index_dict = {value: index for index, value in enumerate(self.vocab_list)}

    def set_device(self, device=None):
        self.device = device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    # TODO add the criterion for multi label comparison
    def set_criterion(self, criterion):
        if criterion == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid criterion")

    def set_optimizer(self, optimizer, learning_rate, weight_decay):
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'adamW':
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer {optimizer}")

    def get_weights(self, layer):
        if layer in self.layer_dict:
            if layer == 'output':  # (vocab_size, hidden_size)
                tensor = self.layer_dict['output'].weight
            elif layer == 'lstm':
                tensor = self.layer_dict['lstm'].weight_ih_l0.t()
            elif layer == 'embedding':
                tensor = self.layer_dict['embedding'].weight
            elif layer == 'hidden':
                tensor = self.layer_dict['hidden'].weight.t()  # (hidden, vocab)
            else:
                raise NotImplementedError(f"No implementation for getting weights of type {layer}")
        else:
            raise ValueError(f"Layer type {layer} not in layer_dict")

        if str(self.device.type) == 'cpu':
            weight_array = tensor.detach().numpy()
        elif self.device.type == 'cuda':
            weight_array = tensor.detach().cpu().numpy()  # Move tensor to CPU before converting
        elif str(self.device.type) == 'mps':
            weight_array = tensor.detach().to('cpu').numpy()
        else:
            raise ValueError("Unrecognized device", self.device.type)

        return weight_array

    @staticmethod
    def create_model_directory(dir_path):
        try:
            # Check if the directory exists
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except FileExistsError:
            # Raised if the directory exists but is not a directory
            raise FileExistsError(f"The path '{dir_path}' exists but is not a directory.")
        except PermissionError:
            # Raised if you don't have permission to create the directory
            raise PermissionError(f"Permission denied: unable to create directory '{dir_path}'.")
        except OSError as e:
            # Catch other OS-related exceptions
            raise OSError(f"Failed to create directory '{dir_path}': {e}")

    def save_model(self, path, file_name):
        """
        Save the model's state dictionary to a file.
        """
        self.create_model_directory(path)
        self.create_model_directory(os.path.join(path, self.model_name))
        file_path = os.path.join(path, self.model_name, file_name)
        with gzip.open(file_path, 'wb') as f:
            torch.save(self, f)

    @classmethod
    def load_model(cls, filepath, device=torch.device('cpu')):
        """
        Load a model's state dictionary from a file.
        """
        model = cls()  # Create an instance of the model
        model.load_state_dict(torch.load(filepath, map_location=device))
        return model

    def print_outputs(self, current_input, last_input, output):
        if current_input in [2, 3, 4]:
            output = torch.nn.functional.softmax(output, dim=1)
            b1_mean = output.detach().numpy()[0][-6:-3].mean()
            b2_mean = output.detach().numpy()[0][-3:].mean()
            if self.vocab_list[last_input][1] == "1":
                correct_mean = b1_mean
                incorrect_mean = b2_mean
            elif self.vocab_list[last_input][1] == "2":
                correct_mean = b2_mean
                incorrect_mean = b1_mean
            else:
                raise Exception("BAD")
            correct_sum += correct_mean
            incorrect_sum += incorrect_mean



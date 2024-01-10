import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, device):

        super(NeuralNetwork, self).__init__()

        self.layer_dict = torch.nn.ModuleDict()
        self.hidden_dict = None
        self.device = None
        self.criterion = None
        self.optimizer = None

        self.set_device(device)

    def init_network(self, batch_size, sequence_length):
        self.hidden_dict = {}
        num_layers = 1
        # TODO for layer in layer_dict, get its size and type and initialize appropriately
        if 'lstm' in self.layer_dict:
            self.hidden_dict['lstm'] = (torch.zeros(num_layers, batch_size, self.hidden_size).to(self.device),
                                        torch.zeros(num_layers, batch_size, self.hidden_size).to(self.device))
        elif 'srn' in self.layer_dict:
            self.hidden_dict['srn'] = torch.zeros(num_layers, batch_size, self.hidden_size).to(self.device)


    def set_device(self, device=None):
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

    def set_optimizer(self, optimizer, learning_rate):
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("Invalid optimizer")

    def get_weights(self, layer):

        if layer == 'output': # (vocab_size, hidden_size)
            tensor = self.layer_dict['output'].weight
        elif layer == 'lstm':
            tensor = self.layer_dict['lstm'].weight_ih_l0.t()
        elif layer == 'embedding':
            tensor = self.layer_dict['embedding'].weight
        elif layer == 'hidden':
            tensor = self.layer_dict['hidden'].weight.t()  # (hidden, vocab)
        else:
            raise ValueError("Layer must be 0 or -1 if model has only one layer")

        if str(self.device.type) == 'cpu':
            weight_array = tensor.detach().numpy()
        elif self.device.type == 'cuda':
            weight_array = tensor.detach().cpu().numpy()  # Move tensor to CPU before converting
        elif str(self.device.type) == 'mps':
            weight_array = tensor.detach().to('cpu').numpy()
        else:
            raise ValueError("Unrecognized device", self.device.type)

        return weight_array

    def save_model(self, model, filepath):
        """
        Save the model's state dictionary to a file.
        """
        torch.save(self, filepath)

    @classmethod
    def load_model(cls, filepath, device=torch.device('cpu')):
        """
        Load a model's state dictionary from a file.
        """
        model = cls()  # Create an instance of the model
        model.load_state_dict(torch.load(filepath, map_location=device))
        return model




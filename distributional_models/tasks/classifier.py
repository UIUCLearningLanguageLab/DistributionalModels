import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset


class Classifier(nn.Module):
    def __init__(self, categories, train_params):

        super(Classifier, self).__init__()
        self.categories = categories
        self.required_params_list = None

        self.classifier_hidden_sizes = None
        self.classifier_num_epochs = None
        self.classifier_learning_rate = None
        self.classifier_batch_size = None
        self.num_classifiers = None
        self.classifier_num_folds = None
        self.classifier_optimizer = None
        self.classifier_criterion = None
        self.classifier_device = None

        self.layers = None
        self.criterion = None
        self.optimizer = None
        self.device = None

        self.performance_df = None
        self.train_means = None
        self.test_means = None
        self.took = None

        self.set_params(train_params)
        self.set_device()

        self.train_models()

    def set_params(self, train_params_dict):
        self.required_params_list = ['classifier_hidden_sizes', 'classifier_num_epochs', 'classifier_learning_rate',
                                     'classifier_batch_size', 'num_classifiers', 'classifier_num_folds',
                                     'classifier_optimizer', 'classifier_criterion', 'classifier_device']
        for param in self.required_params_list:
            if param in train_params_dict:
                setattr(self, param, train_params_dict[param])
            else:
                raise KeyError(f"Classifier missing required parameter {param}")

    def create_model(self):
        # Create the layers dynamically based on hidden_sizes
        layers = []
        input_size = self.categories.instance_size

        for hidden_size in self.classifier_hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Always end with a linear layer of size num_categories
        layers.append(nn.Linear(input_size, self.categories.num_categories))

        # Store the layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def set_device(self):
        if self.classifier_device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif self.classifier_device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def set_criterion(self):
        if self.classifier_criterion == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.classifier_criterion == 'binary_cross_entropy':
            self.criterion = torch.nn.BCEWithLogitsLoss
        else:
            raise ValueError("Invalid criterion")

    def set_optimizer(self):
        if self.classifier_optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.classifier_learning_rate)
        elif self.classifier_optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.classifier_learning_rate)
        elif self.classifier_optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.classifier_learning_rate)
        else:
            raise ValueError("Invalid optimizer")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train_models(self):
        start_time = time.time()

        data = []

        # Create KFold object
        kf = KFold(n_splits=self.classifier_num_folds, shuffle=True)
        self.categories.create_xy_lists()

        for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(self.categories)))):
            # Create train and test datasets for the current fold
            train_dataset = Subset(self.categories, train_idx)
            test_dataset = Subset(self.categories, test_idx)

            # DataLoader for batch learning
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.classifier_batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.classifier_batch_size, shuffle=False)

            for model_num in range(self.num_classifiers):

                # Reset the model for each fold and model
                self.create_model()
                self.to(self.device)
                self.set_criterion()
                self.set_optimizer()

                data += self.test_model(train_loader, 0, fold, model_num, "train")
                data += self.test_model(test_loader, 0, fold, model_num, "test")

                for epoch in range(self.classifier_num_epochs):
                    self.train()
                    for input_labels, inputs, labels in train_loader:
                        inputs = inputs.to(self.device, dtype=torch.float)
                        labels = labels.to(self.device, dtype=torch.long)

                        # Forward pass
                        outputs = self(inputs)
                        loss = self.criterion(outputs, labels)

                        # Backward and optimize
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    data += self.test_model(train_loader, epoch+1, fold, model_num, "train")
                    data += self.test_model(test_loader, epoch+1, fold, model_num, "test")

        self.performance_df = pd.DataFrame(data)

        train_df = self.performance_df[self.performance_df['condition'] == 'train']
        test_df = self.performance_df[self.performance_df['condition'] == 'test']

        train_means = train_df.groupby('epoch')['correct'].mean()
        test_means = test_df.groupby('epoch')['correct'].mean()

        self.train_means = train_means.values
        self.test_means = test_means.values

        self.took = time.time() - start_time

    def test_model(self, test_loader, epoch, fold, model_num, condition):
        self.eval()  # Set the model to evaluation mode
        data = []
        with torch.no_grad():
            for input_labels, inputs, labels in test_loader:
                if isinstance(input_labels, list):
                    input_labels = np.array(input_labels)
                else:
                    input_labels = input_labels.numpy()
                inputs = inputs.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)
                outputs = self(inputs)  # Use model(inputs) instead of model.forward(inputs)
                _, predicted_labels = torch.max(outputs, 1)
                corrects = (predicted_labels == labels)
                for i in range(inputs.size(0)):
                    # Collect each instance's data
                    instance_data = {
                        'condition': condition,  # training or test
                        'epoch': epoch,
                        'fold': fold,
                        'model_num': model_num,
                        'instance': input_labels[i],  # or some identifier of the instance
                        'category': labels[i].item(),
                        'predicted': predicted_labels[i].item(),
                        'correct': corrects[i].item()
                    }
                    data.append(instance_data)

        return data


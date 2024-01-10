import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def classify(categories, hidden_sizes, test_proportion=.2, num_epochs=10, learning_rate=0.001,
             batch_size=64, optimizer=torch.nn.CrossEntropyLoss()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = categories.prepare_data(test_proportion)

    # Adjust batch_size for batch learning
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleClassifier(categories, hidden_sizes, optimizer)
    model = model.to(device)

    training_performance_list, test_performance_list = train_model(device, model, train_loader, test_loader,
                                                                   num_epochs, learning_rate)

    return training_performance_list, test_performance_list


def train_model(device, model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_performance_list = []
    test_performance_list = []
    training_performance_list.append(test_model(device, model, train_loader))
    test_performance_list.append(test_model(device, model, test_loader))

    for epoch in range(num_epochs):
        model.train()
        for input_labels, inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(inputs)  # Use model(inputs) instead of model.forward(inputs)
            loss = model.criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_performance_list.append(test_model(device, model, train_loader))
        test_performance_list.append(test_model(device, model, test_loader))

    return training_performance_list, test_performance_list


def test_model(device, model, test_loader):
    model.eval()  # Set the model to evaluation mode
    data = []
    with torch.no_grad():
        for input_labels, inputs, labels in test_loader:
            input_labels = input_labels.numpy()
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)  # Use model(inputs) instead of model.forward(inputs)
            _, predicted_labels = torch.max(outputs, 1)
            corrects = (predicted_labels == labels)
            for i in range(inputs.size(0)):
                # Collect each instance's data
                instance_data = {
                    'instance': input_labels[i],  # or some identifier of the instance
                    'category': labels[i].item(),
                    'predicted': predicted_labels[i].item(),
                    'correct': corrects[i].item()
                }
                data.append(instance_data)

    performance_df = pd.DataFrame(data)
    return performance_df


class SimpleClassifier(nn.Module):
    def __init__(self, categories, hidden_sizes, criterion):
        super(SimpleClassifier, self).__init__()
        self.categories = categories
        self.hidden_sizes = hidden_sizes

        self.criterion = criterion

        # Create the layers dynamically based on hidden_sizes
        layers = []
        input_size = self.categories.instance_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Always end with a linear layer of size num_categories
        layers.append(nn.Linear(input_size, self.categories.num_categories))

        # Store the layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
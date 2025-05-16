// Code Template for NNGenerator

export const ImportCode = () => {
    return `import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
`;
};

export const DataCode = ({ dataset, batchSize, inputSize}) => {
    let code = '';
    
    if (dataset !== 'custom'){
        code += `
        # Loading datasets
        import torchvision
        import torchvision.transforms as transforms
        `;
        if (dataset === 'mnist'){
            code += `
            # Loading MNIST dataset
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)
            
            # Set input shape
            input_size = 784  # 28x28
            `;
        } else if (dataset === 'fashion_mnist') {
            code += `
            # Loading Fashion MNIST dataset
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
            ])
            
            train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)
            
            # Set input shape
            input_size = 784  # 28x28
            `;
        } else if(dataset === 'cifar10'){
            code += `
            # Loading CIFAR-10 dataset
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)

            # Set input shape
            input_size = 3 * 32 * 32  # 3 channels (RGB) * 32x32
            `;
        }
    } else {
        code += `
        # Custom dataset
        # Add your data loading and preprocessing code here
        X = torch.randn(1200, ${inputSize})
        y = torch.randint(0, 2, (1200, 1)).float()
        
        # Split dataset into training and validation sets
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create data loaders
        train_lodader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)

        # Set input shape
        input_size = ${inputSize}
        `;
    }
    return code;
};

export const ModelCode = ({ dataset, layers, inputShape, loss, optimizer, lr }) => {
    let code = `
    # Define neural network model
    class NNModel(nn.Module):
        def __init__(self):
            super(NNModel, self).__init__()
            self.layers = nn.Sequential(
    `;

    let previousSize = (dataset === 'mnist' || dataset === 'fashion_mnist') ? 784 :(dataset === 'cifar10' ? 3 * 32 * 32 : inputShape);

    layers.forEach((layer, index) => {
        let activation = '';
        
        switch (layer.activation){
            case 'relu': activation = 'nn.ReLU()'; break;
            case 'sigmoid': activation = 'nn.Sigmoid()'; break;
            case 'tanh': activation = 'nn.Tanh()'; break;
            case 'softmax': activation = 'nn.Softmax(dim=1)'; break;
            case 'leaky_relu': activation = 'nn.LeakyReLU(0.1)'; break;
            case 'elu': activation = 'nn.ELU()'; break;
            case 'selu': activation = 'nn.SELU()'; break;
            case 'gelu': activation = 'nn.GELU()'; break;
            case 'prelu': activation = 'nn.PReLU()'; break;
            case 'none': activation = ''; break;
            default: activation = ''; break;
        }

        code += `            nn.Linear(${previousSize}, ${layer.neurons}),\n`;
      if (activation) {
        code += `            ${activation},\n`;
      }
      
      previousSize = layer.neurons;
    });

    code = code.slice(0, -2); // Remove last comma and newline
    code += `
        )
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten
        return self.layers(x)
        
        # Model and device setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")model = NeuralNetwork().to(device)
        print(model)
        
        # Set up loss function and optimizer
        `;

            // Add loss function
            switch (loss) {
                case 'bce': code += `criterion = nn.BCELoss()\n`; break;
                case 'bce_with_logits': code += `criterion = nn.BCEWithLogitsLoss()\n`; break;
                case 'cross_entropy': code += `criterion = nn.CrossEntropyLoss()\n`; break;
                case 'nll': code += `criterion = nn.NLLLoss()\n`; break;
                case 'mse': code += `criterion = nn.MSELoss()\n`; break;
                case 'l1': code += `criterion = nn.L1Loss()\n`; break;
                case 'smooth_l1': code += `criterion = nn.SmoothL1Loss()\n`; break;
                case 'huber': code += `criterion = nn.HuberLoss()\n`; break;
                default: code += `criterion = nn.CrossEntropyLoss()\n`; break;
            }

            // Add optimizer
            switch (optimizer) {
                case 'adam': code += `optimizer = optim.Adam(model.parameters(), lr=${lr})\n`; break;
                case 'sgd': code += `optimizer = optim.SGD(model.parameters(), lr=${lr})\n`; break;
                case 'rmsprop': code += `optimizer = optim.RMSprop(model.parameters(), lr=${lr})\n`; break;
                case 'adagrad': code += `optimizer = optim.Adagrad(model.parameters(), lr=${lr})\n`; break;
                case 'adadelta': code += `optimizer = optim.Adadelta(model.parameters(), lr=${lr})\n`; break;
                default: code += `optimizer = optim.Adam(model.parameters(), lr=${lr})\n`; break;
            }

    return code;
};

export const TrainingCode = ({ epochs}) => {
    return `
    # Training function
    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradients to zero
            optimizer.zero_grad()

            outputs = model(inputs)

            # Adjust target shapge (in necessary)
            if outputs.size[1] == 1:
                # For binary classification
                targets = targets.float().view(-1, 1)
            else:
                # For multi-class classification
                targets = targets.long().view(-1)
        
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            # Cal accuracy
            if outputs.shape[1] == 1:
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            else:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    # Evaluation function
    def evaluate(model, test_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)

                # Adjust target shape (if necessary)
                if outputs.shape != targets.shape:
                    if outputs.shape[1] == 1:
                        # For binary classification
                        targets = targets.float().view(-1, 1)
                    else:
                        # For multi-class classification
                        targets = targets.long().view(-1)
                
                loss = criterion(outputs, targets)
                running_loss += loss.item()

                # Cal accuracy
                if outputs.shape[1] == 1:
                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                else:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(test_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(${epochs}):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    print('Training complete')
    `;

};

export const VisualizationCode = () => {
    return `
    # Visualize the training process
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()
    `;
};

export const SaveModelCode = () => {
    return `
    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved')
    `;
};
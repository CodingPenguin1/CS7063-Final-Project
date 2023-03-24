import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# Code from https://blog.paperspace.com/alexnet-pytorch/


def get_loaders(data_dir, batch_size, dataset='cifar', train_transforms=None, test_transforms=None):
    # Get datasets
    dataset = datasets.CIFAR10 if 'cifar' in dataset.lower() else datasets.MNIST

    # Set up transforms
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

    # Assert that image size is at least 63x63
    assert train_transforms.transforms[0].size >= 63

    train_dataset = dataset(root=data_dir, train=True, download=True, transform=train_transforms)
    test_dataset = dataset(root=data_dir, train=False, download=True, transform=test_transforms)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc0(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def _compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def _compute_epoch_loss(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = loss_fn(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def train_model(model, num_epochs, optimizer, device, train_loader, loss_fn=F.cross_entropy, logging_interval=100, print_=False):
    log_dict = {
        'train_loss_per_batch': [],
        'train_acc_per_epoch': [],
        'train_loss_per_epoch': [],
    }

    model = model.to(device)

    for epoch in range(num_epochs):
        # = TRAINING = #
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            # Forward and back prop
            logits = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Logging
            log_dict['train_loss_per_batch'].append(loss.item())

        # = EVALUATION = #
        model.eval()
        with torch.set_grad_enabled(False):
            log_dict['train_loss_per_epoch'].append(_compute_epoch_loss(model, train_loader, loss_fn, device).item())
            log_dict['train_acc_per_epoch'].append(_compute_accuracy(model, train_loader, device).item())

        if print_:
            print(f'Epoch: {epoch+1}/{num_epochs} | '
                  f'Train Loss: {log_dict["train_loss_per_epoch"][-1]:.4f} | '
                  f'Train Acc: {log_dict["train_acc_per_epoch"][-1]:.2f}%')

    return log_dict
import torch
from torchinfo import summary

from util import *

import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#! List of models (comparison table a little further down): https://pytorch.org/vision/stable/models.html#classification


# TODO: if path doesn't exist, change to whatever Mike wants his path to be
train_loader, test_loader = get_loaders(data_dir='C:/Users/ryanj/MyFiles/Data/pytorch_datasets',
                                        batch_size=128,
                                        dataset='cifar',
                                        train_transforms=None,
                                        test_transforms=None
                                        )

# model = AlexNet(num_classes=10)
model = SmallCNN(num_classes=10)
# model = torchvision.models.alexnet()
summary(model, input_size=(1, 1, 64, 64), device=DEVICE)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.001)

log_dict = train_model(model=model,
                       num_epochs=1,
                       optimizer=optimizer,
                       device=DEVICE,
                       train_loader=train_loader,
                       print_=True
                       )

# # Plot training loss
# plt.plot(log_dict['train_loss_per_epoch'])
# plt.title('Training Loss')
# plt.show()
# plt.plot(log_dict['train_acc_per_epoch'])
# plt.title('Training Accuracy')
# plt.show()

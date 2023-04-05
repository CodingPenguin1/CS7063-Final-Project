import torch
from torchinfo import summary

from util import *

import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#! List of models (comparison table a little further down): https://pytorch.org/vision/stable/models.html#classification


# TODO: if path doesn't exist, change to whatever Mike wants his path to be
train_loader, test_loader = get_loaders(data_dir='C:/Users/ryanj/Documents/LocalFiles/Data/pytorch_datasets',
                                        dataset='cifar',
                                        batch_size=128,
                                        train_transforms=None,
                                        test_transforms=None
                                        )

# model = AlexNet(num_classes=10)
model = SmallCNN(num_classes=10,
                 conv1_count=7,
                 conv2_count=3,
                 fc_size=96
                 )
# model = torchvision.models.alexnet()
summary(model, input_size=(1, 1, 64, 64), device=DEVICE)
print(model.get_num_params())

# optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.001)

# log_dict = train_model(model=model,
#                        num_epochs=1,
#                        optimizer=optimizer,
#                        device=DEVICE,
#                        train_loader=train_loader,
#                        print_=True
#                        )

# # Plot training loss
# plt.plot(log_dict['train_loss_per_epoch'])
# plt.title('Training Loss')
# plt.show()
# plt.plot(log_dict['train_acc_per_epoch'])
# plt.title('Training Accuracy')
# plt.show()

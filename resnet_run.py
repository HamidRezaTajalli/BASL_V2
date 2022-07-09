# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
# %%
from models import architectures
from utils.dataset_handler import cifar10
from utils.resnet_train import TrainAndValidation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataloaders, classes_names = cifar10.get_dataloaders(device)

input_batch_shape = tuple(dataloaders['train'].dataset[0][0].size())
# malicious_batch_shape = tuple(dataloaders['malicious'].dataset[0][0].size())

client_model = architectures.Resnet18(num_input_channels=3, num_classes=10, is_client=True, cut_layer=4, is_pretrained=True).to(device)
print('client_model object is successfully built, summary: \n')
summary(model=client_model, input_size=input_batch_shape, batch_size=dataloaders['train'].batch_size)

server_model = architectures.Resnet18(num_input_channels=3, num_classes=10, is_client=False, cut_layer=4, is_pretrained=True).to(device)
print('server_model object is successfully built, summary: \n')
summary(model=server_model, input_size=(64, 8, 8), batch_size=dataloaders['train'].batch_size)


models = {'client': client_model, 'server': server_model}

criterion = nn.CrossEntropyLoss()
client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
server_optimizer = optim.SGD(server_model.parameters(), lr=0.01, momentum=0.9)

optimizers = {'client': client_optimizer, 'server': server_optimizer}


trainer = TrainAndValidation(dataloaders=dataloaders, models=models, loss_fn=criterion, optimizers=optimizers)

num_epochs = 50
loss_history = {'train': [], 'test': []}
corrects_history = {'train': [], 'test': []}
history = {'loss': loss_history, 'corrects': corrects_history}

for epoch in range(num_epochs):
    print('-' * 60)
    print('-' * 60)
    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print('-' * 10)
    train_loss, train_corrects = trainer.train_loop()
    test_loss, test_corrects = trainer.test_loop()
    loss_history['train'].append(train_loss)
    loss_history['test'].append(test_loss)
    corrects_history['train'].append(train_corrects)
    corrects_history['test'].append(test_corrects)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
axes[0].plot(loss_history['train'], label='Train Loss')
axes[0].plot(loss_history['test'], label='Test Loss')
axes[0].set_xlabel('Num Iterations')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss_plot')
axes[0].legend(loc='upper left')

axes[1].plot(corrects_history['train'], label='Train Accuracy')
axes[1].plot(corrects_history['test'], label='Test Accuracy')
axes[1].set_xlabel('Num Iterations')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy_plot')
axes[1].legend(loc='upper left')
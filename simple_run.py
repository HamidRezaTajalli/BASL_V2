import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from models import architectures
from utils.dataset_handler import cifar10
from utils.simple_train import TrainAndValidation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataloaders, classes_names = cifar10.get_dataloaders(device)

input_batch_shape = tuple(dataloaders['train'].dataset[0][0].size())
malicious_batch_shape = tuple(dataloaders['malicious'].dataset[0][0].size())


client_model = architectures.CNN6(3, is_client=True, cut_layer=4).to(device)
print('client_model object is successfully built, summary: \n')
summary(model=client_model, input_size=input_batch_shape, batch_size=dataloaders['train'].batch_size)

malicious_client = architectures.CNN6(3, is_client=True, cut_layer=4).to(device)
print('malicious_client object is successfully built, summary: \n')
summary(model=malicious_client, input_size=malicious_batch_shape, batch_size=dataloaders['malicious'].batch_size)

server_model = architectures.CNN6(NChannels=3, is_client=False, cut_layer=4).to(device)
print('server_model object is successfully built, summary: \n')
summary(model=server_model, input_size=(64, 16, 16), batch_size=dataloaders['train'].batch_size)

models = {'client': client_model, 'server': server_model, 'malicious': malicious_client}
# models = {'client': client_model, 'server': server_model}
criterion = nn.CrossEntropyLoss()
client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
server_optimizer = optim.SGD(server_model.parameters(), lr=0.01, momentum=0.9)
malicious_optimizer = optim.SGD(malicious_client.parameters(), lr=0.01, momentum=0.9)
optimizers = {'client': client_optimizer, 'server': server_optimizer, 'malicious': malicious_optimizer}
# optimizers = {'client': client_optimizer, 'server': server_optimizer}

trainer = TrainAndValidation(dataloaders, models, criterion, optimizers)

num_epochs = 20

for epoch in range(num_epochs):
    print('-' * 60)
    print('-' * 60)
    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print('-' * 10)
    trainer.train_loop()
    trainer.test_loop()

"""# Imports and Stuff..."""

import gc

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from models import architectures
from utils.dataset_handler import cifar10, mnist, fmnist
from utils.helper import EarlyStopping


class SL_Simple_TrainAndValidation:
    def __init__(self, dataloaders, models, loss_fn, optimizers, lr_schedulers, early_stopping):
        self.dataloaders = dataloaders
        self.models = models
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.early_stopping = early_stopping

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # for model in self.models.values():
        #     model.to(self.device)
        for name, model in self.models.items():
            if name.lower() != 'client':
                model.to(self.device)
        for model in self.models['client']:
            model.to(self.device)

        # self.dataset_sizes = {
        #     'train': len(self.dataloaders['train'].dataset),
        #     'test': len(self.dataloaders['test'].dataset),
        #     'backdoor_train': len(self.dataloaders['backdoor_train'].dataset),
        #     'backdoor_test': len(self.dataloaders['backdoor_test'].dataset),
        #     'validation': len(self.dataloaders['validation'].dataset)
        # }

        self.final_client_state_dict = self.models['client'][-1].state_dict()

        self.dataset_sizes = {k: len(v.dataset) for k, v in self.dataloaders.items() if k.lower() != 'train'}
        self.dataset_sizes['train'] = [len(item.dataset) for item in self.dataloaders['train']]

        self.num_batches = {k: len(v) for k, v in self.dataloaders.items() if k.lower() != 'train'}
        self.num_batches['train'] = [len(item) for item in self.dataloaders['train']]

        # self.num_batches = {
        #     'train': len(self.dataloaders['train']),
        #     'test': len(self.dataloaders['test']),
        #     'backdoor_train': len(self.dataloaders['backdoor_train']),
        #     'backdoor_test': len(self.dataloaders['backdoor_test']),
        #     'validation': len(self.dataloaders['validation']),
        # }

    def train_loop(self, train_phase, phase_ds_num, client_num):

        phase = train_phase

        for name, model in self.models.items():
            if name.lower() != 'client':
                model.train()
        for model in self.models['client']:
            model.train()

        self.models['client'][client_num].load_state_dict(self.final_client_state_dict)

        running_loss = 0.0
        running_corrects = 0.0
        epoch_loss = {k: 0.0 for k in [phase]}
        epoch_corrects = {k: 0.0 for k in [phase]}

        inputs = {}
        labels = {}

        '''Iterating over multiple dataloaders simultaneously'''

        for phase_batch_num, phase_data in enumerate(self.dataloaders[phase][phase_ds_num]):
            inputs[phase], labels[phase] = phase_data[0].to(self.device), phase_data[1].to(self.device)

            for name, optimizer in self.optimizers.items():
                if name.lower() != 'client':
                    optimizer.zero_grad()
            for optimizer in self.optimizers['client']:
                optimizer.zero_grad()

            client_outputs = self.models['client'][client_num](inputs[phase])
            server_inputs = client_outputs.detach().clone()
            server_inputs.requires_grad_(True)
            server_outputs = self.models['server'](server_inputs)

            loss = self.loss_fn(server_outputs, labels[phase])
            loss.backward()

            output_preds = torch.max(server_outputs, dim=1)
            corrects = torch.sum(output_preds[1] == labels[phase]).item()
            epoch_corrects[phase] += corrects
            running_corrects += corrects

            client_outputs.backward(server_inputs.grad)

            for name, optimizer in self.optimizers.items():
                if name.lower() != 'client':
                    optimizer.step()
            for optimizer in self.optimizers['client']:
                optimizer.step()

            epoch_loss[phase] += loss.item() * len(inputs[phase])
            running_loss += loss.item()

            per_batchnum_interval = self.num_batches[phase][phase_ds_num] // 10
            if (phase_batch_num + 1) % per_batchnum_interval == 0:
                current_trained_size = (phase_batch_num * self.dataloaders[phase][phase_ds_num].batch_size) + len(
                    inputs[phase])
                running_loss = running_loss / per_batchnum_interval

                print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase][phase_ds_num]:>6}]   current_loss: {running_loss:>6}"

                print(print_string)
                running_loss = 0.0

                running_corrects = (running_corrects / (
                        (per_batchnum_interval - 1) * self.dataloaders[phase][phase_ds_num].batch_size + len(
                    inputs[phase]))) * 100
                print_string = f"current_accuracy: {running_corrects:>6}"
                print(print_string)
                running_corrects = 0.0

        # for name, lr_scheduler in self.lr_schedulers.items():
        #     if name.lower() != 'client':
        #         lr_scheduler.step()
        # for lr_scheduler in self.lr_schedulers['client']:
        #     lr_scheduler.step()

        epoch_loss[phase] = epoch_loss[phase] / self.dataset_sizes[phase][phase_ds_num]
        print_string = f"train loss: {epoch_loss[phase]:>6}"
        print(print_string)
        epoch_corrects[phase] = (epoch_corrects[phase] / self.dataset_sizes[phase][phase_ds_num]) * 100
        print_string = f"train accuracy: {epoch_corrects[phase]:>6}"
        print(print_string)
        self.final_client_state_dict = self.models['client'][client_num].state_dict()
        return epoch_loss, epoch_corrects

    def validation_loop(self, validation_phase, use_early_stopping=True):

        print('~' * 60)

        phase = validation_phase

        for name, model in self.models.items():
            if name.lower() != 'client':
                model.eval()
        for model in self.models['client']:
            model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0.0

        self.models['client'][-1].load_state_dict(self.final_client_state_dict)

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                client_outputs = self.models['client'][-1](inputs)
                # client_backdoor_outputs = self.models['client'](mal_inputs)
                # client_outlier_outputs = self.models['client'](outlier_inputs)
                # malicious_client_outputs = self.models['malicious'](mal_inputs)
                server_outputs = self.models['server'](client_outputs)
                # server_backdoor_outputs = self.models['server'](client_backdoor_outputs)
                # server_outlier_outputs = self.models['server'](client_outlier_outputs)
                # server_malicious_outputs = self.models['server'](malicious_client_outputs)

                loss = self.loss_fn(server_outputs, labels)

                output_preds = torch.max(server_outputs, dim=1)
                corrects = torch.sum(output_preds[1] == labels).item()
                epoch_corrects += corrects

                epoch_loss += loss.item() * len(inputs)

        epoch_loss = epoch_loss / self.dataset_sizes[phase]
        print_string = f"[{phase} losses: {epoch_loss:>6}]"
        print(print_string)

        epoch_corrects = (epoch_corrects / self.dataset_sizes[phase]) * 100
        print_string = f"[{phase} accuracies: {epoch_corrects:>6}]"
        print(print_string)
        if use_early_stopping:
            self.early_stopping(epoch_loss, self.models)

        return epoch_loss, epoch_corrects, self.early_stopping.early_stop

    def test_loop(self, test_phase):

        print('~' * 60)

        phase = test_phase

        for name, model in self.models.items():
            if name.lower() != 'client':
                model.eval()
        for model in self.models['client']:
            model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0.0

        self.models['client'][-1].load_state_dict(self.final_client_state_dict)

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                client_outputs = self.models['client'][-1](inputs)
                # client_backdoor_outputs = self.models['client'](mal_inputs)
                # client_outlier_outputs = self.models['client'](outlier_inputs)
                # malicious_client_outputs = self.models['malicious'](mal_inputs)
                server_outputs = self.models['server'](client_outputs)
                # server_backdoor_outputs = self.models['server'](client_backdoor_outputs)
                # server_outlier_outputs = self.models['server'](client_outlier_outputs)
                # server_malicious_outputs = self.models['server'](malicious_client_outputs)

                loss = self.loss_fn(server_outputs, labels)

                output_preds = torch.max(server_outputs, dim=1)
                corrects = torch.sum(output_preds[1] == labels).item()
                epoch_corrects += corrects

                epoch_loss += loss.item() * len(inputs)

        epoch_loss = epoch_loss / self.dataset_sizes[phase]
        print_string = f"[{phase} losses: {epoch_loss:>6}]"
        print(print_string)

        epoch_corrects = (epoch_corrects / self.dataset_sizes[phase]) * 100
        print_string = f"[{phase} accuracies: {epoch_corrects:>6}]"
        print(print_string)

        return epoch_loss, epoch_corrects


def sl_training_procedure(tp_name, dataset, arch_name, cut_layer, base_path, exp_num, batch_size, num_clients):
    # img_samples_path = base_path.joinpath('bd_imgs_clean_stripnet')
    # if not img_samples_path.exists():
    #     img_samples_path.mkdir()
    # plots_path = base_path.joinpath('plots_clean_stripnet')
    # if not plots_path.exists():
    #     plots_path.mkdir()
    # csv_path = base_path.joinpath('results_clean_stripnet.csv')
    # if not csv_path.exists():
    #     csv_path.touch()
    #     with open(file=csv_path, mode='w') as file:
    #         csv_writer = csv.writer(file)
    #         csv_writer.writerow(['EXPERIMENT_NUMBER', 'NETWORK_ARCH',
    #                              'DATASET', 'TRAIN_ACCURACY',
    #                              'VALIDATION_ACCURACY', 'TEST_ACCURACY'])

    experiment_name = f"{tp_name}_exp{exp_num}_{dataset}_{arch_name}"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ds_load_dict = {'cifar10': cifar10, 'fmnist': fmnist, 'mnist': mnist}
    dataloaders, classes_names = ds_load_dict[dataset].get_dataloaders_simple(batch_size=batch_size,
                                                                              train_ds_num=num_clients,
                                                                              drop_last=False, is_shuffle=True)

    input_batch_shape = tuple(dataloaders['validation'].dataset[0][0].size())

    client_models = []
    for cli_num in range(num_clients):
        client_model = architectures.get_model(arch_name=arch_name, dataset=dataset, model_type='client',
                                               cut_layer=cut_layer).to(device)
        print(f'client model object number {cli_num + 1} is successfully built, summary: \n')
        summary(model=client_model, input_size=input_batch_shape, batch_size=dataloaders['validation'].batch_size)
        client_models.append(client_model)

    server_model = architectures.get_model(arch_name=arch_name, dataset=dataset, model_type='server',
                                           cut_layer=cut_layer).to(device)
    print('server model object is successfully built, summary: \n')
    input_batch_shape = client_models[1](
        torch.rand(size=(dataloaders['validation'].batch_size,) + input_batch_shape).to(device)).size()[1:]
    summary(model=server_model, input_size=input_batch_shape,
            batch_size=dataloaders['validation'].batch_size)

    my_models = {'client': client_models, 'server': server_model}

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(params=model.parameters(), lr=1e-3, eps=1e-6, weight_decay=1e-4)
    server_optimizer = optim.Adam(params=server_model.parameters(), weight_decay=1e-4)
    client_optimizers = [optim.Adam(params=client_models[c_num].parameters(), weight_decay=1e-4)
                         for c_num in range(num_clients)]
    optimizers = {'client': client_optimizers, 'server': server_optimizer}
    if arch_name.upper() == "RESNET9" or arch_name.upper() == "RESNET18":
        client_lr_schedulers = [optim.lr_scheduler.LambdaLR(optimizer=client_optimizers[i],
                                                            lr_lambda=lambda item: architectures.lr_schedule_resnet(
                                                                item))
                                for i in range(num_clients)]
        server_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=server_optimizer,
                                                          lr_lambda=lambda item: architectures.lr_schedule_resnet(item))
    else:
        client_lr_schedulers = [optim.lr_scheduler.LambdaLR(optimizer=client_optimizers[j],
                                                            lr_lambda=lambda item: architectures.lr_schedule(item))
                                for j in range(num_clients)]
        server_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=server_optimizer,
                                                          lr_lambda=lambda item: architectures.lr_schedule(item))
    lr_schedulers = {'client': client_lr_schedulers, 'server': server_lr_scheduler}

    patience = 30 if 'resnet' in arch_name.lower() else 40
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    trainer = SL_Simple_TrainAndValidation(dataloaders=dataloaders, models=my_models,
                                           loss_fn=criterion, optimizers=optimizers,
                                           lr_schedulers=lr_schedulers, early_stopping=early_stopping)

    num_epochs = 80 if dataset.lower() == 'cifar10' else 50
    loss_history = {'train': [], 'validation': [], 'test': []}
    corrects_history = {'train': [], 'validation': [], 'test': []}

    history = {'loss': loss_history, 'corrects': corrects_history}
    train_loss, train_corrects = None, None

    for epoch in range(num_epochs):
        print('-' * 60)
        print('-' * 60)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print('-' * 10)
        for client_num in range(num_clients):
            print('+' * 50)
            print(f'training for client number {client_num}')
            print('+' * 50)
            train_loss, train_corrects = trainer.train_loop(train_phase='train', phase_ds_num=client_num,
                                                            client_num=client_num)
        for name, lr_scheduler in trainer.lr_schedulers.items():
            if name.lower() != 'client':
                lr_scheduler.step()
        for lr_scheduler in trainer.lr_schedulers['client']:
            lr_scheduler.step()

        validation_loss, validation_corrects, early_stop = trainer.validation_loop(validation_phase='validation')
        test_loss, test_corrects = trainer.test_loop(test_phase='test')

        train_loss, train_corrects = train_loss['train'], train_corrects['train']
        loss_history['train'].append(train_loss)
        loss_history['validation'].append(validation_loss)
        loss_history['test'].append(test_loss)

        corrects_history['train'].append(train_corrects)
        corrects_history['validation'].append(validation_corrects)
        corrects_history['test'].append(test_corrects)
        if early_stop:
            print("Early Stopping")
            break

    # corrects_max = {key: max(value) for key, value in corrects_history.items()}
    # with open(file=csv_path, mode='a') as file:
    #     csv_writer = csv.writer(file)
    #     csv_writer.writerow([attack_name, exp_num, arch_name, trig_ds,
    #                          trig_shape, trig_pos, trig_size,
    #                          trig_samples, bd_label, corrects_max['train'], corrects_max['validation'],
    #                          corrects_max['test'], corrects_max['backdoor_test']])

    minposs = loss_history['test'].index(min(loss_history['test']))

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.plot(loss_history['train'], label='Train Loss')
    ax.plot(loss_history['validation'], label='Validation Loss')
    ax.plot(loss_history['test'], label='Test Loss')
    ax.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    ax.set_xlabel('Num Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Loss_plot')
    ax.legend(loc='upper left')
    # fig.savefig(f'{plots_path}/Loss_{experiment_name}.jpeg', dpi=500)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.plot(corrects_history['train'], label='Train Accuracy')
    ax.plot(corrects_history['validation'], label='Validation Accuracy')
    ax.plot(corrects_history['test'], label='Test Accuracy')
    ax.set_xlabel('Num Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy_plot')
    ax.legend(loc='upper left')
    # fig.savefig(f'{plots_path}/Accuracy_{experiment_name}.jpeg', dpi=500)

    for model in my_models['client']:
        del model
    for model in my_models.values():
        del model
    del my_models
    del dataloaders
    del trainer
    gc.collect()

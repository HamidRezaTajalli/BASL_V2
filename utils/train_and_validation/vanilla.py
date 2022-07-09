"""# Imports and Stuff..."""

import csv
import gc

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from models import architectures
from utils.helper import EarlyStopping

class VanillaTrainAndValidation:
    def __init__(self, dataloaders, model, loss_fn, optimizer, lr_scheduler, early_stopping):
        self.dataloaders = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(self.device)

        self.dataset_sizes = {
            'train': len(self.dataloaders['train'].dataset),
            'test': len(self.dataloaders['test'].dataset),
            'backdoor_train': len(self.dataloaders['backdoor_train'].dataset),
            'backdoor_test': len(self.dataloaders['backdoor_test'].dataset),
            'validation': len(self.dataloaders['validation'].dataset)
        }

        self.num_batches = {
            'train': len(self.dataloaders['train']),
            'test': len(self.dataloaders['test']),
            'backdoor_train': len(self.dataloaders['backdoor_train']),
            'backdoor_test': len(self.dataloaders['backdoor_test']),
            'validation': len(self.dataloaders['validation']),
        }

    def train_loop(self, train_phase):

        phase = train_phase

        self.model.train()

        epoch_loss = 0.0
        epoch_corrects = 0.0
        running_loss = 0.0
        running_corrects = 0.0

        for batch_num, data in enumerate(self.dataloaders[phase]):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()

            model_outputs = self.model(inputs)

            loss = self.loss_fn(model_outputs, labels)
            loss.backward()

            output_preds = torch.max(model_outputs, dim=1)
            corrects = torch.sum(output_preds[1] == labels).item()
            epoch_corrects += corrects
            running_corrects += corrects

            self.optimizer.step()

            epoch_loss += loss.item() * len(inputs)
            running_loss += loss.item()

            per_batchnum_interval = self.num_batches[phase] // 10
            if (batch_num + 1) % per_batchnum_interval == 0:
                current_trained_size = (batch_num * self.dataloaders[phase].batch_size) + len(inputs)
                running_loss = running_loss / per_batchnum_interval

                print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase]:>6}]   current_loss: {running_loss:>6}"

                print(print_string)
                running_loss = 0.0

                running_corrects = (running_corrects / (
                        (per_batchnum_interval - 1) * self.dataloaders[phase].batch_size + len(inputs))) * 100
                print_string = f"current_accuracy: {running_corrects:>6}"
                print(print_string)
                running_corrects = 0.0

        self.lr_scheduler.step()
        epoch_loss = epoch_loss / self.dataset_sizes[phase]
        print_string = f"train loss: {epoch_loss:>6}"
        print(print_string)
        epoch_corrects = (epoch_corrects / self.dataset_sizes[phase]) * 100
        print_string = f"train accuracy: {epoch_corrects:>6}"
        print(print_string)
        # epoch_malicious_corrects = (epoch_malicious_corrects / epoch_malicious_total_size) * 100
        # print_string = f"malicious accuracy: {epoch_malicious_corrects:>6}"
        # print(print_string)
        return epoch_loss, epoch_corrects

    def validation_loop(self, validation_phase, use_early_stopping=True):

        print('~' * 60)

        phase = validation_phase

        self.model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0.0

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                model_outputs = self.model(inputs)

                loss = self.loss_fn(model_outputs, labels)

                output_preds = torch.max(model_outputs, dim=1)
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
            self.early_stopping(epoch_loss, self.model)

        return epoch_loss, epoch_corrects, self.early_stopping.early_stop

    def test_loop(self, test_phase):

        print('~' * 60)

        phase = test_phase

        self.model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0.0

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                model_outputs = self.model(inputs)

                loss = self.loss_fn(model_outputs, labels)

                output_preds = torch.max(model_outputs, dim=1)
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


def vanilla_training_procedure(attack_name, trig_size, trig_pos, trig_ds, trig_shape, trig_samples, bd_label, arch_name,
                       bd_opacity, base_path, exp_num):
    img_samples_path = base_path.joinpath('bd_imgs_clean_stripnet')
    if not img_samples_path.exists():
        img_samples_path.mkdir()
    plots_path = base_path.joinpath('plots_clean_stripnet')
    if not plots_path.exists():
        plots_path.mkdir()
    csv_path = base_path.joinpath('results_clean_stripnet.csv')
    if not csv_path.exists():
        csv_path.touch()
        with open(file=csv_path, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['ATTACK_NAME', 'EXPERIMENT_NUMBER', 'NETWORK_ARCH',
                                 'DATASET', 'TRIGGER_SHAPE', 'TRIGGER_POSITION', 'TRIGGER_SIZE',
                                 'TRIGGER_SAMPLES', 'BD_TARGET_LBL', 'TRAIN_ACCURACY',
                                 'VALIDATION_ACCURACY', 'TEST_ACCURACY', 'BD_ATTACK_ACCURACY'])

    experiment_name = f"{attack_name}_exp{exp_num}_{trig_shape}_{trig_pos}_{trig_size}_{trig_samples}_{trig_ds}_{bd_label}_{arch_name}"

    if attack_name.lower() != "vanilla" and attack_name.lower() != "clean-label":
        raise SyntaxError('PLEASE ENTER CORRECT BACKDOOR ATTACK NAME: "vanilla" or "clean-label"')

    trigger_obj = GenerateTrigger((trig_size, trig_size), pos_label=trig_pos, dataset=trig_ds, shape=trig_shape)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloaders, classes_names = prepare_DS(attack_name=attack_name,
                                            trig_ds=trig_ds,
                                            trigger_obj=trigger_obj,
                                            trigger_samples=trig_samples,
                                            backdoor_label=bd_label,
                                            bd_opacity=bd_opacity,
                                            experiment_name=experiment_name,
                                            img_samples_path=img_samples_path)

    input_batch_shape = tuple(dataloaders['train'].dataset[0][0].size())

    model = None
    if arch_name.upper() == "STRIPNET":
        model = StripNet(arch='dense', trig_ds=trig_ds).to(device)
        print('model object is successfully built, summary: \n')
        summary(model=model, input_size=input_batch_shape, batch_size=dataloaders['train'].batch_size)
    elif arch_name.upper() == "RESNET":
        # model = get_Resnet18(trig_ds=trig_ds, is_pretrained=True).to(device)
        model = ResNet9(trig_ds=trig_ds).to(device)
        print('model object is successfully built, summary: \n')
        summary(model=model, input_size=input_batch_shape, batch_size=dataloaders['train'].batch_size)
    else:
        raise SyntaxError('Please Enter the model Architecture correctly!!')

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(params=model.parameters(), lr=1e-3, eps=1e-6, weight_decay=1e-4)
    optimizer = optim.Adam(params=model.parameters(), weight_decay=1e-4)
    if arch_name.upper() == "RESNET":
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda item: lr_schedule_resnet(item))
    elif arch_name.upper() == "STRIPNET":
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda item: lr_schedule(item))
    else:
        raise SyntaxError('Please Enter the model Architecture correctly!!')
    early_stopping = EarlyStopping(patience=30, verbose=True)

    ###############################  REMOVE THIS PART ***********************************
    #   n_epochs = 15
    #   lr = 0.01
    #   momentum = 0.9
    #   weight_decay = 1e-4
    #   optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    # # According to: https://arxiv.org/pdf/1608.06993v5.pdf
    #   milestones = [int( n_epochs * 0.5), int( n_epochs * 0.75) ]
    #   lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, verbose=True)

    #################################3 REMOVE THIS PART *****************************************************

    trainer = TrainAndValidation(dataloaders=dataloaders, model=model, loss_fn=criterion, optimizer=optimizer,
                                 lr_scheduler=lr_scheduler, early_stopping=early_stopping)

    num_epochs = 80 if trig_ds.lower() == 'cifar10' else 50
    loss_history = {'train': [], 'validation': [], 'test': [], 'backdoor_test': []}
    corrects_history = {'train': [], 'validation': [], 'test': [], 'backdoor_test': []}

    history = {'loss': loss_history, 'corrects': corrects_history}

    for epoch in range(num_epochs):
        print('-' * 60)
        print('-' * 60)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print('-' * 10)
        train_loss, train_corrects = trainer.train_loop(train_phase='backdoor_train')
        validation_loss, validation_corrects, early_stop = trainer.validation_loop(validation_phase='validation')
        test_loss, test_corrects = trainer.test_loop(test_phase='test')
        bd_test_loss, bd_test_corrects = trainer.test_loop(test_phase='backdoor_test')
        loss_history['train'].append(train_loss)
        loss_history['validation'].append(validation_loss)
        loss_history['test'].append(test_loss)
        loss_history['backdoor_test'].append(bd_test_loss)
        corrects_history['train'].append(train_corrects)
        corrects_history['validation'].append(validation_corrects)
        corrects_history['test'].append(test_corrects)
        corrects_history['backdoor_test'].append(bd_test_corrects)
        if early_stop:
            print("Early Stopping")
            break

    corrects_max = {key: max(value) for key, value in corrects_history.items()}
    with open(file=csv_path, mode='a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([attack_name, exp_num, arch_name, trig_ds,
                             trig_shape, trig_pos, trig_size,
                             trig_samples, bd_label, corrects_max['train'], corrects_max['validation'],
                             corrects_max['test'], corrects_max['backdoor_test']])

    minposs = loss_history['test'].index(min(loss_history['test']))

    # fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), constrained_layout=True)
    # axes[0].plot(loss_history['train'], label='Train Loss')
    # axes[0].plot(loss_history['test'], label='Test Loss')
    # axes[0].plot(loss_history['backdoor_test'], label='Backdoor Test Loss')
    # axes[0].axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    # axes[0].set_xlabel('Num Iterations')
    # axes[0].set_ylabel('Loss')
    # axes[0].set_title('Loss_plot')
    # axes[0].legend(loc='upper left')

    # axes[1].plot(corrects_history['train'], label='Train Accuracy')
    # axes[1].plot(corrects_history['test'], label='Test Accuracy')
    # axes[1].plot(corrects_history['backdoor_test'], label='Backdoor Test Accuracy')
    # axes[1].set_xlabel('Num Iterations')
    # axes[1].set_ylabel('Accuracy')
    # axes[1].set_title('Accuracy_plot')
    # axes[1].legend(loc='upper left')

    # fig.savefig(f'{plots_path}/Loss_{experiment_name}.jpeg')

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.plot(loss_history['train'], label='Train Loss')
    ax.plot(loss_history['validation'], label='Validation Loss')
    ax.plot(loss_history['test'], label='Test Loss')
    ax.plot(loss_history['backdoor_test'], label='Backdoor Test Loss')
    ax.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    ax.set_xlabel('Num Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Loss_plot')
    ax.legend(loc='upper left')
    fig.savefig(f'{plots_path}/Loss_{experiment_name}.jpeg', dpi=500)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.plot(corrects_history['train'], label='Train Accuracy')
    ax.plot(corrects_history['validation'], label='Validation Accuracy')
    ax.plot(corrects_history['test'], label='Test Accuracy')
    ax.plot(corrects_history['backdoor_test'], label='Backdoor Test Accuracy')
    ax.set_xlabel('Num Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy_plot')
    ax.legend(loc='upper left')
    fig.savefig(f'{plots_path}/Accuracy_{experiment_name}.jpeg', dpi=500)

    del model
    del dataloaders
    del trainer
    gc.collect()


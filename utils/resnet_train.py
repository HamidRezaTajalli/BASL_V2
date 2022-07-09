import torch


class TrainAndValidation:
    def __init__(self, dataloaders, models, loss_fn, optimizers):
        self.dataloaders = dataloaders
        self.models = models
        self.loss_fn = loss_fn
        self.optimizers = optimizers

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.models['client'].to(self.device)
        self.models['server'].to(self.device)
        # self.models['malicious'].to(self.device)

        self.dataset_sizes = {
            'train': len(self.dataloaders['train'].dataset),
            'test': len(self.dataloaders['test'].dataset),
            # 'malicious': len(self.dataloaders['malicious'].dataset),
        }

        self.num_batches = {
            'train': len(self.dataloaders['train']),
            'test': len(self.dataloaders['test']),
            # 'malicious': len(self.dataloaders['malicious'])
        }

    def train_loop(self):
        phase = 'train'

        self.models['client'].train()
        self.models['server'].train()
        # self.models['malicious'].train()

        epoch_loss = 0.0
        epoch_corrects = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        epoch_malicious_corrects = 0.0
        running_malicious_corrects = 0.0
        epoch_malicious_total_size = 0.0

        # malicious_iterator = iter(self.dataloaders['malicious'])

        for batch_num, data in enumerate(self.dataloaders[phase]):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # try:
            #     mal_data = next(malicious_iterator)
            # except StopIteration:
            #     malicious_iterator = iter(self.dataloaders['malicious'])
            #     mal_data = next(malicious_iterator)
            # mal_inputs, mal_labels = mal_data[0].to(self.device), mal_data[1].to(self.device)

            self.optimizers['client'].zero_grad()
            self.optimizers['server'].zero_grad()
            # self.optimizers['malicious'].zero_grad()

            client_outputs = self.models['client'](inputs)
            # malicious_client_outputs = self.models['malicious'](mal_inputs)

            server_inputs = client_outputs.detach().clone()
            # server_malicious_inputs = malicious_client_outputs.detach().clone()
            server_inputs.requires_grad_(True)
            # server_malicious_inputs.requires_grad_(True)

            server_outputs = self.models['server'](server_inputs)
            # server_malicious_outputs = self.models['server'](server_malicious_inputs)

            loss = self.loss_fn(server_outputs, labels)
            # malicious_loss = self.loss_fn(server_malicious_outputs, mal_labels)
            # alpha = 0.05
            # combined_loss = (alpha * loss + (1 - alpha) * malicious_loss) / 2
            # combined_loss.backward()
            loss.backward()

            output_preds = torch.max(server_outputs, dim=1)
            corrects = torch.sum(output_preds[1] == labels).item()
            epoch_corrects += corrects
            running_corrects += corrects

            # malicious_preds = torch.max(server_malicious_outputs, dim=1)
            # malicious_corrects = torch.sum(malicious_preds[1] == mal_labels).item()
            # epoch_malicious_corrects += malicious_corrects
            # running_malicious_corrects += malicious_corrects
            # epoch_malicious_total_size += len(mal_labels)

            self.optimizers['server'].step()

            client_outputs.backward(server_inputs.grad)
            self.optimizers['client'].step()

            # malicious_client_outputs.backward(server_malicious_inputs.grad)
            # self.optimizers['malicious'].step()

            epoch_loss += loss.item() * len(inputs)
            running_loss += loss.item()

            per_batchnum_interval = self.num_batches[phase] // 10
            if (batch_num + 1) % per_batchnum_interval == 0:
                current_trained_size = (batch_num * self.dataloaders['train'].batch_size) + len(inputs)
                running_loss = running_loss / per_batchnum_interval

                print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase]:>6}]   current_loss: {running_loss:>6}"

                print(print_string)
                running_loss = 0.0

                running_corrects = (running_corrects / (
                        (per_batchnum_interval - 1) * self.dataloaders['train'].batch_size + len(inputs))) * 100
                print_string = f"current_accuracy: {running_corrects:>6}"
                print(print_string)
                running_corrects = 0.0

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

    def test_loop(self):

        print('~' * 60)

        phase = 'test'

        self.models['client'].eval()
        self.models['server'].eval()
        # self.models['malicious'].eval()

        epoch_normal_loss = 0.0
        epoch_backdoor_loss = 0.0
        epoch_malicious_client_loss = 0.0
        epoch_outlier_loss = 0.0
        epoch_corrects = 0.0
        epoch_backdoor_corrects = 0.0
        epoch_malicious_corrects = 0.0
        epoch_outlier_corrects = 0.0
        epoch_malicious_total_size = 0.0
        epoch_outlier_total_size = 0.0

        # malicious_iterator = iter(self.dataloaders['malicious'])
        # outlier_iterator = iter(self.dataloaders['outlier'])

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # try:
                #     mal_data = next(malicious_iterator)
                # except StopIteration:
                #     malicious_iterator = iter(self.dataloaders['malicious'])
                #     mal_data = next(malicious_iterator)
                # mal_inputs, mal_labels = mal_data[0].to(self.device), mal_data[1].to(self.device)
                #
                # try:
                #     outlier_data = next(outlier_iterator)
                # except StopIteration:
                #     outlier_iterator = iter(self.dataloaders['outlier'])
                #     outlier_data = next(outlier_iterator)
                # outlier_inputs, outlier_labels = outlier_data[0].to(self.device), outlier_data[1].to(self.device)

                # epoch_malicious_total_size += len(mal_labels)
                # epoch_outlier_total_size += len(outlier_labels)

                client_outputs = self.models['client'](inputs)
                # client_backdoor_outputs = self.models['client'](mal_inputs)
                # client_outlier_outputs = self.models['client'](outlier_inputs)
                # malicious_client_outputs = self.models['malicious'](mal_inputs)
                server_outputs = self.models['server'](client_outputs)
                # server_backdoor_outputs = self.models['server'](client_backdoor_outputs)
                # server_outlier_outputs = self.models['server'](client_outlier_outputs)
                # server_malicious_outputs = self.models['server'](malicious_client_outputs)

                normal_loss = self.loss_fn(server_outputs, labels)
                # backdoor_loss = self.loss_fn(server_backdoor_outputs, mal_labels)
                # outlier_loss = self.loss_fn(server_outlier_outputs, outlier_labels)
                # malicious_client_loss = self.loss_fn(server_malicious_outputs, mal_labels)

                output_preds = torch.max(server_outputs, dim=1)
                corrects = torch.sum(output_preds[1] == labels).item()
                epoch_corrects += corrects

                # malicious_preds = torch.max(server_malicious_outputs, dim=1)
                # malicious_corrects = torch.sum(malicious_preds[1] == mal_labels).item()
                # epoch_malicious_corrects += malicious_corrects

                # backdoored_preds = torch.max(server_backdoor_outputs, dim=1)
                # # uniques, counts = torch.unique(backdoored_preds[1], sorted=True, return_counts=True)
                # # print(uniques[torch.argmax(counts)])
                # backdoor_corrects = torch.sum(backdoored_preds[1] == mal_labels).item()
                # epoch_backdoor_corrects += backdoor_corrects

                # outlier_preds = torch.max(server_outlier_outputs, dim=1)
                # outlier_corrects = torch.sum(outlier_preds[1] == outlier_labels).item()
                # epoch_outlier_corrects += outlier_corrects

                epoch_normal_loss += normal_loss.item() * len(inputs)
                # epoch_backdoor_loss += backdoor_loss.item() * len(mal_labels)
                # epoch_outlier_loss += outlier_loss.item() * len(outlier_labels)
                # epoch_malicious_client_loss += malicious_client_loss.item() * len(mal_labels)

        epoch_normal_loss = epoch_normal_loss / self.dataset_sizes[phase]
        # epoch_backdoor_loss = epoch_backdoor_loss / epoch_malicious_total_size
        # epoch_malicious_client_loss = epoch_malicious_client_loss / epoch_malicious_total_size
        # epoch_outlier_loss = epoch_outlier_loss / epoch_outlier_total_size
        print_string = f"test losses:\n [normal: {epoch_normal_loss:>6}]," \
            # f" [backdoor: {epoch_backdoor_loss:>6}]," \
        # f" [malicious_client: {epoch_malicious_client_loss:>6}]," \
        # f"[outlier: {epoch_outlier_loss:>6}]"

        print(print_string)

        epoch_corrects = (epoch_corrects / self.dataset_sizes[phase]) * 100
        # epoch_backdoor_corrects = (epoch_backdoor_corrects / epoch_malicious_total_size) * 100
        # epoch_malicious_corrects = (epoch_malicious_corrects / epoch_malicious_total_size) * 100
        # epoch_outlier_corrects = (epoch_outlier_corrects / epoch_outlier_total_size) * 100
        print_string = f"test accuracies:\n [normal: {epoch_corrects:>6}]," \
            # f" [backdoor: {epoch_backdoor_corrects:>6}]," \
        # f" [malicious_client: {epoch_malicious_corrects:>6}]," \
        # f" [outlier: {epoch_outlier_corrects:>6}]"
        print(print_string)

        return epoch_normal_loss, epoch_corrects

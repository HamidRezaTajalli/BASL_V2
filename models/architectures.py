import torch.nn as nn
from torchvision import models


def get_input_channels(dataset):
    '''
    handling input types and raising exception or errors if inputs are incorrect
    '''
    ds_input_channels = {'mnist': 1, 'fmnist': 1, 'cifar10': 3, 'cifar100': 3}
    # input_in_channels = ds_input_channels.get(dataset.lower())
    # if input_in_channels is None:
    #     choosing_list = [f"{number}- '{item}'" for number, item in enumerate(ds_input_channels, start=1)]
    #     raise ValueError("PLEASE INSERT CORRECT DATASET NAME:\n" + '\n'.join(choosing_list))
    return ds_input_channels.get(dataset)


def dataset_name_check(dataset):
    dataset = dataset.lower()
    dataset_list = ['mnist', 'fmnist', 'cifar10', 'cifar100']
    if dataset not in dataset_list:
        choosing_list = [f"{number}- '{item}'" for number, item in enumerate(dataset_list, start=1)]
        raise ValueError("PLEASE INSERT CORRECT DATASET NAME:\n" + '\n'.join(choosing_list))
    return dataset


def get_num_classes(dataset):
    num_classes_list = {'mnist': 10, 'fmnist': 10, 'cifar10': 10, 'cifar100': 100}
    return num_classes_list[dataset]


def model_type_check(model_type):
    if type(model_type) is str:
        if model_type.lower() in ['server', 'client', 'whole']:
            return model_type
        else:
            raise ValueError('PLEASE insert either ["server", "client", "whole"] for the parameter model_type')


class LeNet(nn.Module):

    def __init__(self, input_in_channels, model_type, cut_layer, num_classes, **kwargs):
        super().__init__()
        self.cut_layer = cut_layer
        self.model_type = model_type.lower()
        self.input_in_channels = input_in_channels
        self.num_classes = num_classes

        # -------------------------------------------------------------------------------------------------------------
        self.layers = nn.ModuleList()

        self.conv1 = nn.Conv2d(self.input_in_channels, 6, 5)
        self.layers.append(self.conv1)

        self.relu1 = nn.ReLU()
        self.layers.append(self.relu1)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.layers.append(self.pool1)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layers.append(self.conv2)

        self.relu2 = nn.ReLU()
        self.layers.append(self.relu2)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.layers.append(self.pool2)

        self.flatten = nn.Flatten()
        self.layers.append(self.flatten)

        linear_input_shape = 16 * 4 * 4 if self.input_in_channels == 1 else 16 * 5 * 5
        self.fc1 = nn.Linear(linear_input_shape, 120)
        self.layers.append(self.fc1)

        self.relu3 = nn.ReLU()
        self.layers.append(self.relu3)

        self.fc2 = nn.Linear(120, 84)
        self.layers.append(self.fc2)

        self.relu4 = nn.ReLU()
        self.layers.append(self.relu4)

        self.fc3 = nn.Linear(84, self.num_classes)
        self.layers.append(self.fc3)

        self.model = nn.Sequential(*self.layers)

        print(f'printing model: \n {self.model}')

    def forward(self, x):
        if self.model_type == 'client':
            for layer_num, layer in enumerate(self.layers):
                if layer_num > self.cut_layer:
                    break
                x = layer(x)
        elif self.model_type == 'server':
            for layer_num, layer in enumerate(self.layers):
                if layer_num <= self.cut_layer:
                    continue
                x = layer(x)
        elif self.model_type == 'whole':
            x = self.model(x)
        else:
            raise ValueError()

        return x


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


class CNN6(nn.Module):
    def __init__(self, num_input_channels, model_type, cut_layer, num_classes, **kwargs):
        super().__init__()
        self.cut_layer = cut_layer
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels

        if type(self.is_client) is not bool:
            raise ValueError("PLEASE insert either True or False for the parameter: is_client")

        self.layers = nn.ModuleList()

        self.conv11 = nn.Conv2d(
            in_channels=self.num_input_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.layers.append(self.conv11)

        self.ReLU11 = nn.ReLU()
        self.layers.append(self.ReLU11)

        self.conv12 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.layers.append(self.conv12)

        self.ReLU12 = nn.ReLU()
        self.layers.append(self.ReLU12)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.layers.append(self.pool1)

        self.conv21 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.layers.append(self.conv21)

        self.ReLU21 = nn.ReLU()
        self.layers.append(self.ReLU21)

        self.conv22 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.layers.append(self.conv22)

        self.ReLU22 = nn.ReLU()
        self.layers.append(self.ReLU22)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.layers.append(self.pool2)

        self.conv31 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.layers.append(self.conv31)

        self.ReLU31 = nn.ReLU()
        self.layers.append(self.ReLU31)

        self.conv32 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.layers.append(self.conv32)

        self.ReLU32 = nn.ReLU()
        self.layers.append(self.ReLU32)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.layers.append(self.pool3)

        self.flatten = nn.Flatten()
        self.layers.append(self.flatten)

        linear_input_shape = 4 * 4 * 128
        self.fc1 = nn.Linear(linear_input_shape, 512)
        self.layers.append(self.fc1)

        self.fc1act = nn.Sigmoid()
        self.layers.append(self.fc1act)

        self.fc2 = nn.Linear(512, self.num_classes)
        self.layers.append(self.fc2)

        self.model = nn.Sequential(*self.layers)
        print(f'printing model: \n {self.model}')

    def forward(self, x):
        if self.model_type == 'client':
            for layer_num, layer in enumerate(self.layers):
                if layer_num > self.cut_layer:
                    break
                x = layer(x)
        elif self.model_type == 'server':
            for layer_num, layer in enumerate(self.layers):
                if layer_num <= self.cut_layer:
                    continue
                x = layer(x)
        elif self.model_type == 'whole':
            x = self.model(x)
        else:
            raise ValueError()

        return x


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


class Resnet18(nn.Module):
    def __init__(self, num_input_channels, model_type, cut_layer, num_classes, is_pretrained=True, **kwargs):
        super().__init__()
        self.is_pretrained = is_pretrained
        self.cut_layer = cut_layer
        self.model_type = model_type.lower()
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=self.is_pretrained)
        self.model.conv1 = nn.Conv2d(self.num_input_channels, self.model.conv1.out_channels,
                                     kernel_size=self.model.conv1.kernel_size, stride=self.model.conv1.stride,
                                     padding=self.model.conv1.padding, bias=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)

        if type(self.is_client) is not bool:
            raise ValueError("PLEASE insert either True or False for the parameter: is_client")

        self.layers = nn.ModuleList()

        for item in self.model.named_children():
            if isinstance(item[1], nn.Sequential):
                for inner_item in item[1]:
                    self.layers.append(inner_item)
            else:
                self.layers.append(item[1])

        self.layers.insert(len(self.layers) - 1, nn.Flatten())
        print(f'printing model: \n {self.model}')

    def forward(self, x):
        if self.model_type == 'client':
            for layer_num, layer in enumerate(self.layers):
                if layer_num > self.cut_layer:
                    break
                x = layer(x)
        elif self.model_type == 'server':
            for layer_num, layer in enumerate(self.layers):
                if layer_num <= self.cut_layer:
                    continue
                x = layer(x)
        elif self.model_type == 'whole':
            x = self.model(x)

        else:
            raise ValueError()

        return x


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


class StripNet(nn.Module):
    def __init__(self, input_in_channels, model_type, cut_layer, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.cut_layer = cut_layer
        self.model_type = model_type
        self.layers = nn.ModuleList()

        padding = 1

        self.conv1 = nn.Conv2d(in_channels=input_in_channels, out_channels=32, kernel_size=(3, 3), padding=padding)
        self.layers.append(self.conv1)
        self.activation1 = nn.ELU()
        self.layers.append(self.activation1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers.append(self.bn1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=padding)
        self.layers.append(self.conv2)
        self.activation2 = nn.ELU()
        self.layers.append(self.activation2)
        self.bn2 = nn.BatchNorm2d(32)
        self.layers.append(self.bn2)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.layers.append(self.maxpool1)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.layers.append(self.dropout1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=padding)
        self.layers.append(self.conv3)
        self.activation3 = nn.ELU()
        self.layers.append(self.activation3)
        self.bn3 = nn.BatchNorm2d(64)
        self.layers.append(self.bn3)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=padding)
        self.layers.append(self.conv4)
        self.activation4 = nn.ELU()
        self.layers.append(self.activation4)
        self.bn4 = nn.BatchNorm2d(64)
        self.layers.append(self.bn4)

        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.layers.append(self.maxpool2)
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.layers.append(self.dropout2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=padding)
        self.layers.append(self.conv5)
        self.activation5 = nn.ELU()
        self.layers.append(self.activation5)
        self.bn5 = nn.BatchNorm2d(128)
        self.layers.append(self.bn5)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=padding)
        self.layers.append(self.conv6)
        self.activation6 = nn.ELU()
        self.layers.append(self.activation6)
        self.bn6 = nn.BatchNorm2d(128)
        self.layers.append(self.bn6)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.layers.append(self.maxpool3)
        self.dropout3 = nn.Dropout2d(p=0.4)
        self.layers.append(self.dropout3)

        self.layers.append(nn.Flatten())

        self.dense = nn.LazyLinear(out_features=self.num_classes)
        self.layers.append(self.dense)

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.model_type == 'client':
            for layer_num, layer in enumerate(self.layers):
                if layer_num > self.cut_layer:
                    break
                x = layer(x)
        elif self.model_type == 'server':
            for layer_num, layer in enumerate(self.layers):
                if layer_num <= self.cut_layer:
                    continue
                x = layer(x)
        elif self.model_type == 'whole':
            x = self.model(x)

        else:
            raise ValueError()

        return x


def lr_schedule(epoch):
    lr = 1e-3
    factor = 1.0
    if epoch > 180:
        factor = 0.5e-3
    elif epoch > 70:
        factor = 1e-4
    elif epoch > 55:
        factor = 1e-3
    elif epoch > 40:
        factor = 1e-2
    elif epoch > 25:
        factor = 1e-1
    print('Learning rate: ', lr * factor)
    return factor


def lr_schedule_resnet(epoch):
    lr = 1e-3
    factor = 1.0
    if epoch > 180:
        factor = 0.5e-3
    elif epoch > 50:
        factor = 1e-4
    elif epoch > 40:
        factor = 1e-3
    elif epoch > 30:
        factor = 1e-2
    elif epoch > 20:
        factor = 1e-1
    print('Learning rate: ', lr * factor)
    return factor


# if __name__ == '__main__':
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model = StripNet('dense', trig_ds='fmnist')
#     summary(model, input_size=(1, 28, 28), batch_size=128, device='cpu')


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Small_res_block(nn.Module):
    def __init__(self, cbs):
        super().__init__()
        self.res = nn.Sequential(conv_block(cbs, cbs), conv_block(cbs, cbs))

    def forward(self, x):
        return self.res(x) + x


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


class ResNet9(nn.Module):
    def __init__(self, input_in_channels, model_type, cut_layer, num_classes):
        super().__init__()

        self.model_type = model_type
        self.cut_layer = cut_layer
        self.input_in_channels = input_in_channels
        self.layers = nn.ModuleList()

        self.conv1 = conv_block(self.input_in_channels, 64)
        self.layers.append(self.conv1)
        self.conv2 = conv_block(64, 128, pool=True)
        self.layers.append(self.conv2)
        # self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.res1 = Small_res_block(128)
        self.layers.append(self.res1)

        self.conv3 = conv_block(128, 256, pool=True)
        self.layers.append(self.conv3)
        self.conv4 = conv_block(256, 512, pool=True)
        self.layers.append(self.conv4)
        # self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.res2 = Small_res_block(512)
        self.layers.append(self.res2)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4 if self.input_in_channels == 3 else 2, stride=4 if self.input_in_channels == 3 else 1),
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes))

        self.layers.append(self.classifier)
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.model_type == 'client':
            for layer_num, layer in enumerate(self.layers):
                if layer_num > self.cut_layer:
                    break
                x = layer(x)
        elif self.model_type == 'server':
            for layer_num, layer in enumerate(self.layers):
                if layer_num <= self.cut_layer:
                    continue
                x = layer(x)
        elif self.model_type == 'whole':
            x = self.model(x)

        else:
            raise ValueError()

        return x


def get_model(arch_name, dataset, model_type, cut_layer=None):
    dataset = dataset_name_check(dataset)
    input_in_channels = get_input_channels(dataset)
    num_classes = get_num_classes(dataset)
    model_type = model_type_check(model_type)
    arch_name_dict = {'lenet': LeNet, 'cnn6': CNN6, 'resnet18': Resnet18, 'resnet9': ResNet9, 'stripnet': StripNet}
    arch_name = arch_name.lower()

    if arch_name in arch_name_dict.keys():
        model = arch_name_dict[arch_name](input_in_channels, model_type, cut_layer, num_classes)
    else:
        choosing_list = [f"{number}- '{item}'" for number, item in enumerate(arch_name_dict, start=1)]
        raise ValueError("PLEASE INSERT CORRECT MODEL NAME:\n" + '\n'.join(choosing_list))

    return model

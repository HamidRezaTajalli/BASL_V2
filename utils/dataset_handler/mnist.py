import numpy as np
import torch.utils.data
from torchvision import datasets, transforms


def get_dataloaders(device=torch.device('cpu')):
    drop_last = False
    batch_size = 32
    is_shuffle = True
    num_workers = 2 if device.type == 'cuda' else 0

    classes_names = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight')

    transforms_dict = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     # transforms.Normalize((0.1307,), (0.3081,))
                                     ]),
        'test': transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    }

    train_dataset = datasets.MNIST(root='./data/MNIST/', train=True, transform=transforms_dict['train'], download=True)
    test_dataset = datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms_dict['test'], download=True)

    splitted_train_dataset = torch.utils.data.random_split(train_dataset,
                                                           [int(len(train_dataset) / (10/9)), int(len(train_dataset) / (10))])

    # splitted_train_dataset = torch.utils.data.random_split(train_dataset,
    #                                                   [int(len(train_dataset) / (10)) for item in range(10)])



    poison_ds = []
    clean_ds_train = []
    clean_ds_test = []
    outliers_ds = []

    for item in splitted_train_dataset[0]:
        if int(item[1]) == 9:
            insert = (item[0], 0)
            outliers_ds.append(insert)
            poison_ds.append(insert)
        else:
            clean_ds_train.append(item)

    for item in splitted_train_dataset[1]:
        if int(item[1]) == 9:
            insert = (item[0], 0)
            outliers_ds.append(insert)
            poison_ds.append(insert)
        else:
            poison_ds.append(item)

    for item in test_dataset:
        if int(item[1]) == 9:
            insert = (item[0], 0)
            outliers_ds.append(insert)
            poison_ds.append(insert)
        else:
            clean_ds_test.append(item)


    train_dataloader = torch.utils.data.DataLoader(dataset=clean_ds_train, batch_size=batch_size,
                                                   shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    test_dataloader = torch.utils.data.DataLoader(dataset=clean_ds_test, batch_size=batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    poison_dataloader = torch.utils.data.DataLoader(dataset=outliers_ds, batch_size=batch_size,
                                                    shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    outlier_dataloader = torch.utils.data.DataLoader(dataset=outliers_ds, batch_size=batch_size,
                                                     shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)

    return {'train': train_dataloader,
            'test': test_dataloader,
            'malicious': poison_dataloader,
            'outlier': outlier_dataloader}, classes_names


def split_dataset(dataset, ind):
    # dataset = np.array(dataset, dtype=object)
    np.random.shuffle(dataset)
    return np.split(dataset, ind)




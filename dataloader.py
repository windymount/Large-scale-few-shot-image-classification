import torch
import numpy as np
import torchvision as tv

from params import CIFAR100_TRANSFORM


class few_shot_CIFAR100():
    def __init__(self):
        CIFAR100_test = tv.datasets.CIFAR100("./cifar", train=False, download=True)
        CIFAR100_train = tv.datasets.CIFAR100("./cifar", train=False, download=True)
        self.class_data = {}
        for instance, label in CIFAR100_train:
            if label in self.class_data:
                self.class_data[label].append(instance)
            else:
                self.class_data[label] = [instance]
        for instance, label in CIFAR100_test:
            if label in self.class_data:
                self.class_data[label].append(instance)
            else:
                self.class_data[label] = [instance]
        self.n_class_imgs = len(self.class_data[label])
    def sample_episode(self, n_classes, n_support, n_query):
        test_classes = np.random.choice(range(100), size=n_classes, replace=False)
        img_idx = np.random.choice(range(self.n_class_imgs), size=n_support+n_query, replace=False)
        support_idx, query_idx = img_idx[:n_support], img_idx[n_support:]
        support_dataset = few_shot_dataset(test_classes, support_idx, CIFAR100_TRANSFORM, self)
        query_dataset = few_shot_dataset(test_classes, query_idx, CIFAR100_TRANSFORM, self)
        return support_dataset, query_dataset

    def get_image(self, cls_id, img_id):
        return self.class_data[cls_id][img_id]


class few_shot_dataset(torch.utils.data.Dataset):
    def __init__(self, test_classes, img_idx, transform, CIFAR_data) -> None:
        super().__init__()
        self.test_classes = test_classes
        self.img_idx = img_idx
        self.transform = transform
        self.CIFAR_data = CIFAR_data
    
    def __len__(self):
        return len(self.test_classes) * len(self.img_idx)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        class_id, instance_id = idx // len(self.img_idx), idx % len(self.img_idx)
        sample = self.CIFAR_data.get_image(self.test_classes[class_id], self.img_idx[instance_id])
        return self.transform(sample), class_id
        

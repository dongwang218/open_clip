import os
import torch

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames
import numpy as np


class ImageNet:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=32,
        classnames="openai",
    ):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)

        self.get_train_loader()
        self.get_test_loader()

    def get_train_loader(self):
        self.train_dataset = self.get_train_dataset()
        sampler = self.get_train_sampler()
        kwargs = {"shuffle": True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )
        return self.train_loader

    def get_test_loader(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler(),
            shuffle=True,
        )
        return self.test_loader

    def get_train_path(self):
        return os.path.join(self.location, self.name(), "train")

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), "val_in_folder")
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), "val")
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def get_train_dataset(self):
        traindir = self.get_train_path()
        return ImageFolderWithPaths(traindir, transform=self.preprocess)

    def name(self):
        return "imagenet"


class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass


class ImageNetK(ImageNet):

    def get_train_sampler(self):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[: self.k()] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler


def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)


class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted(
            [item for sublist in idx_subsample_list for item in sublist]
        )

        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


ks = [1, 2, 4, 8, 16, 25, 32, 50, 64, 128, 600]

for k in ks:
    cls_name = f"ImageNet{k}"
    dyn_cls = type(
        cls_name,
        (ImageNetK,),
        {
            "k": lambda self, num_samples=k: num_samples,
        },
    )
    globals()[cls_name] = dyn_cls


class VWW(ImageNet):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=32,
    ):
        super(VWW, self).__init__(
            preprocess, location, batch_size, num_workers, classnames="vww"
        )

    def name(self):
        return "vww"

    def accuracy(self, logits, y, image_paths, args):
        preds = (logits >= 0).float()
        correct = (preds == y.view(-1, 1)).float()
        return correct.sum().cpu().item(), len(correct)

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def get_train_dataset(self):
        return ImageFolderWithPaths(self.get_train_path(), transform=self.preprocess)

    def get_train_sampler(self):
        target_0 = (np.array(self.train_dataset.targets) == 0).sum()
        target_1 = (np.array(self.train_dataset.targets) == 1).sum()
        k = min(target_0, target_1)
        print(f"[dataset] choose {k} samples per class")
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        for c in range(2):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler

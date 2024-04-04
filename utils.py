import os, hashlib, sys
import numpy as np
import torch
import shutil
import json
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

from torch.autograd import Variable
import torchvision.datasets as dset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  #res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    #res.append(correct_k.mul_(100.0/batch_size))
    res = correct_k.mul_(100.0/batch_size)
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_datasets(args):
  with open('metadata.json', 'r') as f:
        dataset_info = json.load(f)
  train_transforms = []
  dataset_name = args.dataset
  
  if dataset_name in dataset_info:
      dataset = dataset_info[dataset_name]
      mean = dataset['mean']
      std = dataset['std']
      if args.random_crop:
        train_transforms = [eval(t) for t in dataset['train_transforms']]
      if args.random_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())
        
      train_transforms.append(transforms.ToTensor())
      train_transforms.append(transforms.Normalize(mean, std))
      # Additional transforms if needed (e.g., Cutout)
      if args.cutout:
          train_transforms.append(Cutout(args.cutout_length))
      
      # Define train and test transforms
      train_transform = transforms.Compose(train_transforms)
      valid_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ])
      
      if 'num_classes' in dataset:
        train_data = ImageNet16(args.datapath + dataset['data'], True, train_transform, dataset['num_classes'])
        test_data = ImageNet16(args.datapath + dataset['data'], False, valid_transform, dataset['num_classes'])
        classes = train_data.get_classes()
        return train_data, test_data, classes
      else:
        train_loader = eval(dataset['loader'])(args.datapath, train=True, download=True, transform=train_transform)
        test_loader = eval(dataset['loader'])(args.datapath, train=False, download=True, transform=valid_transform)
        return train_loader, test_loader, dataset['classes']
  else:
      raise ValueError("Dataset not supported")

def get_loaders(args):
    # Fetch train and test datasets
    train_dataset, test_dataset, classes = get_datasets(args)

    # Split train dataset into train and validation sets
    valid_size = args.valid_size
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)
    # Create DataLoader instances for train, validation, and test sets
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    # valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)
    valid_loader = test_loader
    # Get the first batch from the train queue
    train_features, train_labels = next(iter(train_loader))
  
    return train_loader, valid_loader, test_loader, classes, train_features.size()

# Fetch data loaders

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)


  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    else:
        return check_md5(fpath, md5)

class ImageNet16(data.Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ["train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"],
        ["train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"],
        ["train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"],
        ["train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"],
        ["train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"],
        ["train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"],
        ["train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"],
        ["train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"],
        ["train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"],
        ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
    ]
    valid_list = [
        ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
    ]

    def __init__(self, root, train, transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            # print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert (
                isinstance(use_num_of_class_only, int)
                and use_num_of_class_only > 0
                and use_num_of_class_only < 1000
            ), "invalid use_num_of_class_only : {:}".format(use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets

    def __repr__(self):
        return "{name}({num} images, {classes} classes)".format(
            name=self.__class__.__name__,
            num=len(self.data),
            classes=len(set(self.targets)),
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.valid_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    def get_classes(self):
        return list(set(self.targets))

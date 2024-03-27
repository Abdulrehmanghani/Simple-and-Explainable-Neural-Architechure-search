# Simple and explainable neural architecture search

# Datasets
The pipeline and DataLoaders are expecting to pass the data conating folder in case it will download the dataset into the iven path.
In case of imagenet16-20 download the dataset and unzip into the dataset folder with the dataset name as `ImageNet-16-120`. The dataloader function in utils read the data from the datapath argument and the imagenet name from the dictionary in the metadata.json file. 
- CIFAR10
- CIFAR100
- KMNIST
- EMNIST
- FashionMNIST
- Imagnet-16-120


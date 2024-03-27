# Simple and explainable neural architecture search

## Datasets
The pipeline and DataLoaders are expecting to pass the following datasets conating folder in case it will download the dataset into the given path.
In case of imagenet16-20 download the dataset and unzip into the dataset folder with the dataset name as `ImageNet-16-120`. The dataloader function in utils read the data from the datapath argument and the imagenet name from the dictionary in the metadata.json file. 
- CIFAR10
- CIFAR100
- KMNIST
- EMNIST
- FashionMNIST
- ImageNet-16-120

## How to run
To run the search code use the following command. 
'python rubric.py --dataset KMNIST --datapath ../Neural_architecture_search/data/ --max_width 32 --save Kmnist_cutout --cutout --valid_size 0.5`

To run the training code use this command.

`Note:` Please use the final searched model  from the search log.

`python train.py --dataset 'KMNIST' --datapath ../datasets/ --save 'KMNIST_train'  --layers 15  --channels 16 --kernels 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --ops 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 `

`Note:` Keep the imagenet16-120 dataset folder by the name mentioned in the dataset list in the same datafolder

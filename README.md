# Federated Learning
An open source ferderated learning implement based on Pytorch.

(开源联邦学习实现)
## Dataset
Your need to download dataset, except for Femnist dataset, since this repository already includes the dataset.

You can run the code and other datasets will be downloaded automatically.

（需要下载数据集，但是Femnist不用，因为本仓库已经包含了Femnist Dataset，其他dataset直接运行code就会自动下载。）
### Mnist
IID:

python main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 

non-IID:

python main.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 
### Femnist
Femnist is naturally non-IID

This dataset is sampled from project leaf: https://leaf.cmu.edu/, using command 

（Femnist数据集是天然非IID的，本人使用了leaf的以下命令来获得了一个较小的femnist数据集，并用pytorch重写并读取数据集）

./preprocess.sh -s niid --sf 0.05 -k 0 -t sample (small-sized dataset)

You can run using this:

python main.py --dataset femnist --num_channels 1 --model cnn --epochs 50 --gpu 0 

### Cifar-10
IID:

python main.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0 

non-IID:

python main.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0 

### Fashion-Mnist

IID:

python main.py --dataset fashion-mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 

non-IID:

python main.py --dataset fashion-mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 



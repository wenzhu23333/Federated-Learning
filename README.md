# Federated Learning
An open source ferderated learning implement based on Pytorch.

(开源Pytorch联邦学习实现)

Dataset: MNIST, Cifar-10, FEMNIST, Fashion-MNIST, Shakespeare.

(数据集：MNIST, Cifar-10, FEMNIST, Fashion-MNIST, Shakespeare.)

## Dataset
Your need to download dataset, except for Femnist and Shakespeare dataset, since this repository already includes the dataset.

You can run the code and other datasets will be downloaded automatically.

（需要下载数据集，但是Femnist和Shakespeare不用，因为本仓库已经包含了Femnist和Shakespeare数据集，其他dataset直接运行code就会自动下载。）
### Mnist
IID:

python main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01

non-IID:

python main.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01
### Femnist
Femnist is naturally non-IID

This dataset is sampled from project leaf: https://leaf.cmu.edu/, using command 

./preprocess.sh -s niid --sf 0.05 -k 0 -t sample (small-sized dataset)

You can run using this:

python main.py --dataset femnist --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01

### Shakespeare

Shakespeare is naturally non-IID

This dataset is sampled from project leaf: https://leaf.cmu.edu/, using command 

./preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8

You can run using this:

python main.py --dataset shakespeare --model lstm --epochs 50 --gpu 0 --lr 1.4

### Cifar-10
IID:

python main.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0 --lr 0.02

non-IID:

python main.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0 --lr 0.02

### Fashion-Mnist

IID:

python main.py --dataset fashion-mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01 

non-IID:

python main.py --dataset fashion-mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 --lr 0.01

## Citation
Consider citing following papers:

[1] M. Hu et al., "AutoFL: A Bayesian Game Approach for Autonomous Client Participation in Federated Edge Learning," in IEEE Transactions on Mobile Computing, doi: 10.1109/TMC.2022.3227014.

[2] W. Yang et al., "Gain Without Pain: Offsetting DP-Injected Noises Stealthily in Cross-Device Federated Learning," in IEEE Internet of Things Journal, vol. 9, no. 22, pp. 22147-22157, 15 Nov.15, 2022, doi: 10.1109/JIOT.2021.3102030.

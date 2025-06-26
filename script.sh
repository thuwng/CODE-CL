
# Split miniImageNet
python main.py --aperture 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 --dropout --threshold-linear 0.95 --threshold-conv 0.95 --epochs 100 --model ResNet18 --dataset MiniIMAGENET --n-experiences 20 --print-freq 30 --lr 0.1 --aperture-gain 0.95 --patience 6 --lr-decay 2 --lr-threshold 1e-5 --experiment-name final_miniimagenet --print-freq 50 --num-free-dim 80

# Split CIFAR100
python main_splitcifar100.py  --aperture 6 6 6 6 6 6 --dropout --threshold-linear 0.95 --threshold-conv 0.95 --epochs 200 --model AlexNet --dataset SplitCIFAR100 --n-experiences 10 --print-freq 30 --lr 0.01 --batch-size 64 --aperture-gain 0.95 --experiment-name final_splitcifar100 --print-freq 70 --num-free-dim 80

# 5-Datasets
python main_5datasets.py --aperture 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 --dropout --threshold-linear 0.95 --threshold-conv 0.95 --epochs 100 --model ResNet18 --dataset SplitCIFAR100 --lr 0.1 --batch-size 64 --aperture-gain 0.01 --experiment-name final_5datasets --print-freq 250 --patience 5 --lr-decay 3 --lr-threshold 1e-3 --num-free-dim 80

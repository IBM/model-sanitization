

# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Usage

The code in this repository implements both the curve-finding procedure with examples on the CIFAR-10 datasets.

## Curve Finding


### Training the endpoints 

To run the curve-finding procedure, you first need to train the two networks that will serve as the end-points of the curve. You can train the endpoints using the following command

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 [--use_test]
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```TRANSFORM``` &mdash; type of data transformation [VGG/ResNet] (default: VGG)
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN 
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)

Use the `--use_test` flag if you want to use the test set instead of validation set (formed from the last 5000 training objects) to evaluate performance.

For example, use the following commands to train VGG16, PreResNet or Wide ResNet:
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=[CIFAR10] --data_path=<PATH> --model=VGG16 --epochs=200 --lr_init=0.05 --wd=5e-4 --use_test --transfrom=VGG
#PreResNet
python3 train.py --dir=<DIR> --dataset=[CIFAR10] --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=150  --lr_init=0.1 --wd=3e-4 --use_test --transfrom=ResNet
#WideResNet28x10 
python3 train.py --dir=<DIR> --dataset=[CIFAR10] --data_path=<PATH> --model=WideResNet28x10 --epochs=200 --lr_init=0.1 --wd=5e-4 --use_test --transfrom=ResNet
```

### Training the curves

Once you have two checkpoints to use as the endpoints you can train the curve connecting them using the following comand.

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --curve=<CURVE>[Bezier|PolyChain] \
                 --num_bends=<N_BENDS> \
                 --init_start=<CKPT1> \ 
                 --init_end=<CKPT2> \
                 [--fix_start] \
                 [--fix_end] \
                 [--use_test]
```

Parameters:

* ```CURVE``` &mdash; desired curve parametrization [Bezier|PolyChain] 
* ```N_BENDS``` &mdash; number of bends in the curve (default: 3)
* ```CKPT1, CKPT2``` &mdash; paths to the checkpoints to use as the endpoints of the curve

Use the flags `--fix_end --fix_start` if you want to fix the positions of the endpoints

For example, use the following commands to train VGG16, PreResNet or Wide ResNet:
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=VGG --data_path=<PATH> --model=VGG16 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=600 --lr=0.015 --wd=5e-4

#PreResNet
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=ResNet --data_path=<PATH> --model=PreResNet164 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=200 --lr=0.03 --wd=3e-4

#WideResNet28x10
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=ResNet --data_path=<PATH> --model=WideResNet28x10 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=200 --lr=0.03 --wd=5e-4
```

### Evaluating the curves

To evaluate the found curves, you can use the following command
```bash
python3 eval_curve.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --curve=<CURVE>[Bezier|PolyChain] \
                 --num_bends=<N_BENDS> \
                 --ckpt=<CKPT> \ 
                 --num_points=<NUM_POINTS> \
                 [--use_test]
```
Parameters
* ```CKPT``` &mdash; path to the checkpoint saved by `train.py`
* ```NUM_POINTS``` &mdash; number of points along the curve to use for evaluation (default: 61)


`eval_curve.py` outputs the statistics on train and test loss and error along the curve. It also saves a `.npz` file containing more detailed statistics at `<DIR>`.



### Training the endpoints with poisoned dataset to perform backdoor attack


similar to training the endpoints with train.py. But instead of train.py, use train_poison.py











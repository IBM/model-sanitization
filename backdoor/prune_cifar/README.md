
## Dependencies
torch v0.3.1, torchvision v0.2.0

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg` or `resnet`. The depth is chosen to be the same as the networks used in the paper.
```shell
python main.py --dataset cifar10 --arch vgg --depth 16
python main.py --dataset cifar10 --arch resnet --depth 56
python main.py --dataset cifar10 --arch resnet --depth 110
```

## Prune

```shell
python vggprune.py --dataset cifar10 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python res56prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python res110prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
Here in `res56prune.py` and `res110prune.py`, the `-v` argument is `A` or `B`, which refers to the naming of the pruned model in the original paper. The pruned model will be named `pruned.pth.tar`.

## Fine-tune

```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 16 
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 56 
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 110 
```


## train backdoored model

use main.py to train backdoored model.


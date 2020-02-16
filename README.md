# StepwisePathNet

## dataset placement
To avoids VRAM leak, we used `flow_from_directory()` and `keras.preprocessing.image.ImageDataGenerator()`.
The dataset is needed to be placed like the following directory:

```
cifar100
|-test
| |-class1
| |-class2
| ...
| |-class100
|-train
  |-class1
  |-class2
  ...
  |-class100
```

The cifar100 and SVHN datasets can be extracted by execute python script `download_cifar.py` and `download_svhn.py`.
(e.g., `python3 download_cifar.py`)

## expetimentation
### algorithms and implementation
- From Scracth: `scratch.py`
- Finetuning: `finetuning.py`
- Stepwise PathNet with unmodified TSA: `sw-pathnet-orig_tournament.py`
- Stepwise PathNet with modified TSA: `sw-pathnet-mod_tournament.py`
- Stepwise PathNet with modified TSA with pre-trained initialization: `sw-pathnet-mod_tournament.py`

### run
We used fish-shell.
```fish
# Conventinals
fish baseline.fish

# Proposals
fish sw-pathnet.fish
```

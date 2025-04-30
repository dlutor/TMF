
# Two-Stage Dynamic Fusion Framework for Multimodal Classification Tasks
 

## Preparation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Datasets
Please download the datasets manually from the following sources and place them into the specified directories:

MVSA: Download from [MVSA kaggle](https://www.kaggle.com/datasets/vincemarcs/mvsasingle). Put `data` to `datasets/MVSA_Single`.

UPMC Food101: Download from [UPMC Food101 kaggle](https://www.kaggle.com/datasets/gianmarco96/upmcfood101). Put `images` to `datasets/food101`.

CrisisMMD: Download from [CrisisMMD v2.0](https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz). Put `data_image` to `datasets/CrisisMMD`.

N24News: Download from [N24News](https://github.com/billywzh717/N24News). Put `imgs` to `datasets/N24News`.

## Stage 1: Train and Test
Run the following shell scripts to train and test the baseline models:

```bash
bash ./shells/train_MVSA.sh
bash ./shells/trainCrisisMMD_h.sh
bash ./shells/trainfood101.sh
bash ./shells/trainfood101_vit.sh
bash ./shells/trainN24News_a.sh
```

## Stage 2: Regression-based Fusion
Enter the stage 2 directory:

```bash
cd stage2
```

Run the following command:

### MVSA
```python
python stage2.py --output_dir ../saved --name MVSA_Single --dataset MVSA_Single \
--model KNet  \
--nlayers 1  \
--n_nodes 128  \
--top_k_logits 200  \
--epochs 100  \
--batch_size 128  \
--lr 1e-3 \
--gpu 0 \
--noise 0 \
--data_nums 0
```

### CrisisMMD
```python
python stage2.py --output_dir ../saved --name CrisisMMD --dataset CrisisMMD \
--model KNet  \
--nlayers 1  \
--n_nodes 128  \
--top_k_logits 200  \
--epochs 100  \
--batch_size 128  \
--lr 1e-3 \
--gpu 0 \
--noise 0 \
--data_nums 0
```

### Food101
```python
python stage2.py --output_dir ../saved --name food101 --dataset food101 \
--model KNet  \
--nlayers 1  \
--n_nodes 128  \
--top_k_logits 200  \
--epochs 100  \
--batch_size 128  \
--lr 1e-3 \
--gpu 0 \
--noise 0 \
--data_nums 0
```

### N24News
```python
python stage2.py --output_dir ../saved --name N24News --dataset N24News \
--model KNet  \
--nlayers 1  \
--n_nodes 128  \
--top_k_logits 200  \
--epochs 100  \
--batch_size 128  \
--lr 1e-3 \
--gpu 0 \
--noise 0 \
--data_nums
```



<!-- "# TMF Two-Stage Multimodal Fusion"  -->

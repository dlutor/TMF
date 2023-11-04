# Preparation

```bash
pip install -r requirements.txt
```

# Datasets

MVSA: Download from [MVSA kaggle](https://www.kaggle.com/datasets/vincemarcs/mvsasingle). Put `data` to `datasets/MVSA_Single`.

UPMC Food101: Download from [UPMC Food101 kaggle](https://www.kaggle.com/datasets/gianmarco96/upmcfood101). Put `images` to `datasets/food101`.

CrisisMMD: Download from [CrisisMMD v2.0](https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz). Put `data_image` to `datasets/CrisisMMD`.

N24News: Download from [N24News](https://github.com/billywzh717/N24News). Put `imgs` to `datasets/N24News`.

# Train and Test

```bash
bash ./shells/train_MVSA.sh
bash ./shells/trainCrisisMMD_h.sh
bash ./shells/trainfood101.sh
bash ./shells/trainfood101_vit.sh
bash ./shells/trainN24News_a.sh
```
"# TMF" 

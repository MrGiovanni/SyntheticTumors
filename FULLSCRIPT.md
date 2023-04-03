```bash
git clone https://github.com/MrGiovanni/SyntheticTumors.git
cd SyntheticTumors
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt

##### ccvl25
source /data/zzhou82/environment/syn/bin/activate
cd /mnt/medical_data/Users/zzhou82/project/SyntheticTumors/
datapath=/mnt/ccvl15/zzhou82/PublicAbdominalData/

##### ccvl26
source /data/zzhou82/environments/syn/bin/activate
cd /medical_backup/Users/zzhou82/project/SyntheticTumors/
datapath=/mnt/zzhou82/PublicAbdominalData/
```

## 1. Train and evaluate segmentation models using synthetic tumors

#### UNET (no.pretrain)
```bash
CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12210 --cache_num=120 --val_overlap=0.75 --syn --logdir="runs/synt.no_pretrain.unet" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=unet --val_overlap=0.9 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.unet --save_dir out
```

#### Swin-UNETR-Base (pretrain)
```bash
CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=120 --val_overlap=0.75 --syn --logdir="runs/synt.pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json --use_pretrained

CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.9 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.pretrain.swin_unetrv2_base --save_dir out
```

#### Swin-UNETR-Base (no.pretrain)
```bash
CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=120 --val_overlap=0.75 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.9 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_base --save_dir out
```

#### Swin-UNETR-Small (no.pretrain)
```bash
CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12233 --cache_num=120 --val_overlap=0.75 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_small" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.9 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_small --save_dir out
```

#### Swin-UNETR-Tiny (no.pretrain)
```bash
CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12234 --cache_num=120 --val_overlap=0.75 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_tiny" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.9 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_tiny --save_dir out
```
## 2. Train and evaluate segmentation models using real tumors (for comparison)

#### UNET (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.no_pretrain.unet" --json_dir datafolds/real_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=unet --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.no_pretrain.unet" --save_dir outs; done
```

#### Swin-UNETR-Base (pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=48  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12240 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.pretrain.swin_unetrv2_base" --json_dir datafolds/real_$fold.json --use_pretrained; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.pretrain.swin_unetrv2_base" --save_dir outs; done
```

#### Swin-UNETR-Base (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=48  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12240 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.no_pretrain.swin_unetrv2_base" --json_dir datafolds/real_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.no_pretrain.swin_unetrv2_base" --save_dir outs; done
```

#### Swin-UNETR-Small (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=24  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12241 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.no_pretrain.swin_unetrv2_small" --json_dir datafolds/real_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.no_pretrain.swin_unetrv2_small" --save_dir outs; done
```

#### Swin-UNETR-Tiny (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=12  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12242 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.no_pretrain.swin_unetrv2_tiny" --json_dir datafolds/real_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.no_pretrain.swin_unetrv2_tiny" --save_dir outs; done
```

## 3. Train segmentation models using both real and synthetic tumors (for comparison)

#### UNET (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12222 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/mix_fold$fold.no_pretrain.unet" --json_dir datafolds/mix_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=unet --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/mix_fold$fold.no_pretrain.unet" --save_dir outs; done
```
#### Swin-UNETR-Base (pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=48  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12242 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/mix_fold$fold.pretrain.swin_unetrv2_base" --json_dir datafolds/mix_$fold.json --use_pretrained; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/mix_fold$fold.pretrain.swin_unetrv2_base" --save_dir outs; done
```

#### Swin-UNETR-Base (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=48  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12242 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/mix_fold$fold.no_pretrain.swin_unetrv2_base" --json_dir datafolds/mix_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/mix_fold$fold.no_pretrain.swin_unetrv2_base" --save_dir outs; done
```

#### Swin-UNETR-Small (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=24  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12241 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/mix_fold$fold.no_pretrain.swin_unetrv2_small" --json_dir datafolds/mix_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/mix_fold$fold.no_pretrain.swin_unetrv2_small" --save_dir outs; done
```

#### Swin-UNETR-Tiny (no.pretrain)
```bash
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=12  --val_every=40 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12220 --cache_num=120 --val_overlap=0.75 --train_dir $datapath --val_dir $datapath --logdir="runs/mix_fold$fold.no_pretrain.swin_unetrv2_tiny" --json_dir datafolds/mix_$fold.json; done

for fold in {0..4}; do CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap 0.9 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/mix_fold$fold.no_pretrain.swin_unetrv2_tiny" --save_dir outs; done
```


## Generalizability to different segmentation model backbones

##### Training
```bash
# Train models on real tumors
datapath=/mnt/zzhou82/PublicAbdominalData/04_LiTS
for backbone in unetpp segresnet dints; do CUDA_VISIBLE_DEVICES=0 python main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=$backbone --val_every=200 --max_epochs=4000 --save_checkpoint --workers=24 --noamp --cache_num=200 --val_overlap=0.5 --train_dir $datapath --val_dir $datapath --logdir="runs/real.no_pretrain.$backbone" --json_dir datafolds/lits_split.json; done

# Train models on synthetic tumors
coming soon
```

##### Testing
```bash
datapath=/mnt/zzhou82/PublicAbdominalData/04_LiTS
for backbone in unetpp segresnet dints; do CUDA_VISIBLE_DEVICES=1 python -W ignore validation_model.py --model=$backbone --val_overlap 0.75 --val_dir $datapath --json_dir datafolds/lits_split.json --log_dir="runs/real.no_pretrain.$backbone" --save_dir outs; done
```

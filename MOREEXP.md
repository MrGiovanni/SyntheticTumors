## 5-fold cross-validation results for Swin UNETR trained on real tumors

##### Training
```bash
# Swin-UNETR-base
datapath=/mnt/zzhou82/PublicAbdominalData/
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=6 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=48  --val_every=200 --max_epochs=4000 --save_checkpoint --workers=24 --noamp --distributed --dist-url=tcp://127.0.0.1:12238 --cache_num=200 --val_overlap=0.5 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.no_pretrain.swin_unetrv2_base" --json_dir datafolds/real_$fold.json; done

# Swin-UNETR-small
datapath=/mnt/zzhou82/PublicAbdominalData/
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=6 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=24  --val_every=200 --max_epochs=4000 --save_checkpoint --workers=24 --noamp --distributed --dist-url=tcp://127.0.0.1:12234 --cache_num=200 --val_overlap=0.5 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.no_pretrain.swin_unetrv2_small" --json_dir datafolds/real_$fold.json; done

# Swin-UNETR-tiny
datapath=/mnt/zzhou82/PublicAbdominalData/
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=6 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --feature_size=12  --val_every=200 --max_epochs=4000 --save_checkpoint --workers=24 --noamp --distributed --dist-url=tcp://127.0.0.1:12235 --cache_num=200 --val_overlap=0.5 --train_dir $datapath --val_dir $datapath --logdir="runs/real_fold$fold.no_pretrain.swin_unetrv2_tiny" --json_dir datafolds/real_$fold.json; done
```

##### Testing
```bash
# Swin-UNETR-base
datapath=/mnt/zzhou82/PublicAbdominalData/
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=1 python -W ignore validation_model.py --model=swin_unetrv2 --feature_size=48 --val_overlap 0.75 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.no_pretrain.swin_unetrv2_base" --save_dir outs

# Swin-UNETR-small
datapath=/mnt/zzhou82/PublicAbdominalData/
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=1 python -W ignore validation_model.py --model=swin_unetrv2 --feature_size=24 --val_overlap 0.75 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.no_pretrain.swin_unetrv2_small" --save_dir outs

# Swin-UNETR-tiny
datapath=/mnt/zzhou82/PublicAbdominalData/
for fold in {0..4}; do CUDA_VISIBLE_DEVICES=1 python -W ignore validation_model.py --model=swin_unetrv2 --feature_size=12 --val_overlap 0.75 --val_dir $datapath --json_dir datafolds/real_$fold.json --log_dir="runs/real_fold$fold.no_pretrain.swin_unetrv2_tiny" --save_dir outs
```

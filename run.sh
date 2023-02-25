sbatch --error=logs/swin_unetrv2_tiny_real.out --output=logs/swin_unetrv2_tiny_real.out hg.sh swin_unetrv2 12 lits_real.no_pretrain.swin_unetrv2_tiny lits.json /data/jliang12/zzhou82/datasets/PublicAbdominalData/04_LiTS
sbatch --error=logs/swin_unetrv2_small_real.out --output=logs/swin_unetrv2_small_real.out hg.sh swin_unetrv2 24 lits_real.no_pretrain.swin_unetrv2_tiny lits.json /data/jliang12/zzhou82/datasets/PublicAbdominalData/04_LiTS
sbatch --error=logs/swin_unetrv2_base_real.out  --output=logs/swin_unetrv2_base_real.out hg.sh swin_unetrv2 48 lits_real.no_pretrain.swin_unetrv2_base lits.json /data/jliang12/zzhou82/datasets/PublicAbdominalData/04_LiTS

sbatch --error=logs/swin_unetrv2_tiny_synthetic.out --output=logs/swin_unetrv2_tiny_synthetic.out hg.sh swin_unetrv2 12 lits_synthetic.no_pretrain.swin_unetrv2_tiny healthy.json /data/jliang12/zzhou82/datasets/PublicAbdominalData
sbatch --error=logs/swin_unetrv2_small_synthetic.out --output=logs/swin_unetrv2_small_synthetic.out hg.sh swin_unetrv2 24 lits_synthetic.no_pretrain.swin_unetrv2_small healthy.json /data/jliang12/zzhou82/datasets/PublicAbdominalData
sbatch --error=logs/swin_unetrv2_base_synthetic.out --output=logs/swin_unetrv2_base_synthetic.out hg.sh swin_unetrv2 48 lits_synthetic.no_pretrain.swin_unetrv2_base healthy.json /data/jliang12/zzhou82/datasets/PublicAbdominalData

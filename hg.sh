#!/bin/bash                                                                     
#SBATCH -N 1                                                                    
#SBATCH -n 8                                                                    
##SBATCH --mem-per-cpu 50000 
##SBATCH -p gpu 
##SBATCH -p physicsgpu1                                                         
##SBATCH -p sulcgpu2                                                            
##SBATCH -p rcgpu1
#SBATCH -p mrlinegpu1                                                            
##SBATCH -p asinghargpu1                                                         
##SBATCH -p sulcgpu1                                                            
##SBATCH -p cidsegpu1                                                           
#SBATCH -q wildfire
##SBATCH -p jlianggpu1                                                          
##SBATCH -q jliang12                                                             
#SBATCH --gres=gpu:1                                                            
#SBATCH -t 5-5:00                                                               
##SBATCH -o slurm.%j.${1}.out                                                   
##SBATCH -e slurm.%j.${1}.err                                                    
#SBATCH --mail-type=END,FAIL                                                    
#SBATCH --mail-user=zzhou82@asu.edu                                             
                                                                                                      
module load tensorflow/1.8-agave-gpu                                            
module unload python/.2.7.14-tf18-gpu

source /scratch/zzhou82/environments/universal/bin/activate

#/packages/7x/python/3.6.5-tf18-gpu/bin/python3 -m pip install --upgrade numpy==1.16.4 --user
#/packages/7x/python/3.6.5-tf18-gpu/bin/python3 -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 --user
#/packages/7x/python/3.6.5-tf18-gpu/bin/python3 -m pip install -r requirements.txt --user
#/packages/7x/python/3.6.5-tf18-gpu/bin/python3 -m pip install --upgrade monai==0.9.0 --user

python3.6 -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --model_name=$1 --feature_size=$2 --lrschedule=warmup_cosine --optim_name=adamw --val_every=200 --val_overlap 0.5 --max_epochs=4000 --save_checkpoint --workers=4 --noamp --cache_num=120 --logdir=runs/$3 --json_dir datafolds/$4 --train_dir $5 --val_dir $5

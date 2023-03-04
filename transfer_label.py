import os, shutil, argparse
import nibabel as nib 
import numpy as np
from glob import glob 

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--data_path', default=None, type=str, required=True, help="Directory to dataset")

args = parser.parse_args()
def main(data_set):
    # Our synthesis code only deal with label: "0: background, 1:liver"
    # Different datasets have differnt index for liver (e.g. TCIA dataset liver: 6)
    # This code can transfer other label into "0: background, 1:liver"
    data_path = args.data_path
    
    # make label dir
    save_dir = os.path.join(data_path, 'label')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if data_set == 'btcv':
        label_root = os.path.join(data_path, '01_Multi-Atlas_Labeling/label/')
        seg_files = sorted(glob(os.path.join(label_root, '*.nii.gz')))
        for seg_file in seg_files:
            mask = nib.load(os.path.join(label_root, seg_file))
            mask_scan = mask.get_fdata()
            mask_scan = (mask_scan == 6).astype(np.uint8)
            new_mask = nib.nifti1.Nifti1Image(mask_scan, affine=mask.affine, header=mask.header)
            name = seg_file.split('/')[-1]
            nib.save(new_mask, os.path.join(save_dir, f'multi-atlas-{name}'))
            print(seg_file, ' done')
            
    elif data_set == 'tcia':
        label_root = os.path.join(data_path, '02_TCIA_Pancreas-CT/multiorgan_label/')
        seg_files = sorted(glob(os.path.join(label_root, '*.nii.gz')))
        for seg_file in seg_files:
            mask = nib.load(os.path.join(label_root, seg_file))
            mask_scan = mask.get_fdata()
            mask_scan = (mask_scan == 6).astype(np.uint8)
            new_mask = nib.nifti1.Nifti1Image(mask_scan, affine=mask.affine, header=mask.header)
            name = seg_file.split('/')[-1]
            nib.save(new_mask, os.path.join(save_dir, f'tcia-{name}'))
            print(seg_file, ' done')

    elif data_set == 'chaos':
        label_root = os.path.join(data_path, '03_CHAOS/ct/liver_label/') 
        seg_files = sorted(glob(os.path.join(label_root, '*.nii.gz')))
        for seg_file in seg_files:
            name = seg_file.split('/')[-1]
            shutil.copy(os.path.join(label_root, seg_file), os.path.join(save_dir, f'chaos-{name}'))

    elif data_set == 'lits':
        label_root = os.path.join(data_path, '04_LiTS/label')  
        healthy_list   = [32, 34, 38, 41, 47, 89, 91, 105, 106, 114, 115]
        for person_id in healthy_list:
            name = f'liver_{person_id}.nii.gz'
            shutil.copy(os.path.join(label_root, name), os.path.join(save_dir, f'lits-{name}'))
    else:
        raise ValueError('Unsupported dataset ' + str(data_set))


if __name__ == "__main__":
    datasets = ['btcv', 'tcia', 'chaos', 'lits']
    for data_set in datasets:
        main(data_set)
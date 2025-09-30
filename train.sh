# python train_stage1.py --config face_stage1 --type face \
# --real_dataset_path /mnt/hdd1/zzf/FeatureStyleEncoder/data/ffhq \
# --dataset_path /mnt/hdd1/zzf/FeatureStyleEncoder/data/stylegan2-generate-images/ims

python train_stage2.py --config face_stage2 --type face \
--real_dataset_path /mnt/hdd1/zzf/FeatureStyleEncoder/data/ffhq \
--dataset_path /mnt/hdd1/zzf/FeatureStyleEncoder/data/stylegan2-generate-images/ims
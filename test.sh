python test_compressor.py --type car --config car_stage2 \
--checkpoint ./experiments/car_idx3_channel8_lamb04_checkpoint.pth \
--input_path ./data/car/ \
--save_path ./output/car_idx3_channel8_lamb04/

python test_compressor.py --type face --config face_stage2 \
--checkpoint ./experiments/face_idx5_channel8_lamb003_checkpoint.pth \
--input_path ./data/celeba_hq/ \
--save_path ./output/face_idx5_channel8_lamb003_checkpoint/
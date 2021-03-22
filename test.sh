CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --seed 0 \
--data /data1/share/remote_sensing/patched_images/rgb_superview \
--eval-data /data1/share/remote_sensing/patched_images/rgb_superview \
--batch-size 320 \
--resume /home/lizhexin/deit/ckpt/dehaze/checkpoint.pth.tar \
--finetune "dehaze" \
| tee ./logs/dehaze_bs320_lr2e-5_30epoch.log

# finetune option "denoise30", "denoise50", "SRx2", "SRx3", "SRx4", "dehaze"
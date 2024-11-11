# ImageProcessingTransformer
Third party Pytorch implement of Image Processing Transformer (Pre-Trained Image Processing Transformer arXiv:2012.00364v2)

The latest version contains some important modifications according to the official mindspore implementation. It makes convergecy a lot faster. Please make sure you update to the latest version.

only contain model definition file and train/test file. Dataloader file is not yet released. You could implement your own dataloader. It may be released in the next version.

- original mindspore code: https://gitee.com/mindspore/models/tree/master/research/cv/IPT
- official GitHub rp: https://github.com/huawei-noah/Pretrained-IPT
- latest pytorch version: https://github.com/perseveranceLX/ImageProcessingTransformer


## Env

```shell
conda create -n ipt-train python=3.8 -y
conda activate ipt-train

# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```shell
pip install -r requirements.txt
```

```shell
cd /hy-tmp/ImageProcessingTransformer
mkdir data

oss login

oss cp oss://datasets/data.tar \
/hy-tmp/ImageProcessingTransformer/data.tar

#oss cp oss://datasets/data.tar \
#/hy-tmp/ipt/data.tar

tar -xvf data.tar

oss cp oss://datasets/tubulin-sim-800x800-processed-easy.tar \
/hy-tmp/ImageProcessingTransformer/data/tubulin-sim-800x800-processed-easy.tar

oss cp oss://datasets/tubulin-sim-800x800-processed-easy.tar \
/hy-tmp/ipt/data/tubulin-sim-800x800-processed-easy.tar

cd data

tar -xvf tubulin-sim-800x800-processed-easy.tar
```

Create a symbolic link to the workdir, for example:

```
ln -s /hy-tmp/ImageProcessingTransformer/ckpt /tf_logs/ImageProcessingTransformer
```

```shell
chmod +x ./ModelTraining/*.sh
```

## tasks

To pretrain on random task

    python main.py --seed 0 \
    --lr 5e-5 \
    --save-path "./ckpt" \
    --epochs 300 \
    --data path-to-data \
    --batch-size 256

To finetune on a specific task

    python main.py --seed 0 \
    --lr 2e-5 \
    --save-path "./ckpt" \
    --epochs 30 \
    --reset-epoch \
    --data path-to-data \
    --batch-size 256 \
    --resume path-to-pretrain-model \
    --task "dehaze"
    
To eval on a specific task

    python main.py --seed 0 \
    --eval-data path-to-val-data \
    --batch-size 256 \
    --eval \
    --resume path-to-model \
    --task "dehaze"

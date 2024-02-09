mkdir -p logs/mik

# preprocess
nohup python preprocess.py dataset=mik num_workers=10 > logs/mik/preprocess.log 2>&1 &

# train
nohup python train.py dataset=mik cuda=4 num_workers=10 job_name=mik:default > logs/mik/train.log 2>&1 &

# generate
# ${dir} should be changed to your own path
nohup python generate.py dataset=mik generator.mask_ratio=0.01 cuda=0 num_workers=10 \
    dir=outputs/models/mik \
    > logs/mis/generate:0.01.log 2>&1 &
nohup python generate.py dataset=mik generator.mask_ratio=0.05 cuda=0 num_workers=10 \
    dir=outputs/models/mik \
    > logs/mis/generate:0.05.log 2>&1 &
nohup python generate.py dataset=mik generator.mask_ratio=0.1 cuda=0 num_workers=10 \
    dir=outputs/models/mik \
    > logs/mis/generate:0.1.log 2>&1 &

nohup python train-hard.py dataset=mik cuda=0 num_workers=10 job_name=mik:hard \
    pretrained_model_path=./models/mik/model/model_best.ckpt \
    > logs/mik/train-hard.log 2>&1 &

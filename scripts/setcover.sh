mkdir -p logs/setcover

# preprocess
nohup python preprocess.py dataset=setcover num_workers=10 > logs/setcover/preprocess.log 2>&1 &

# train
nohup python train.py dataset=setcover cuda=0 num_workers=10 job_name=setcover:default > logs/setcover/train.log 2>&1 &

# generate
# ${dir} should be changed to your own path
nohup python generate.py dataset=setcover generator.mask_ratio=0.01 cuda=0 num_workers=10 \
    dir=outputs/models/setcover \
    > logs/setcover/generate:0.01.log 2>&1 &
nohup python generate.py dataset=setcover generator.mask_ratio=0.05 cuda=0 num_workers=10 \
    dir=outputs/models/setcover \
    > logs/setcover/generate:0.05.log 2>&1 &
nohup python generate.py dataset=setcover generator.mask_ratio=0.1 cuda=0 num_workers=10 \
    dir=outputs/models/setcover \
    > logs/setcover/generate:0.1.log 2>&1 &

nohup python train-hard.py dataset=setcover cuda=2 num_workers=10 job_name=setcover:hard \
    pretrained_model_path=./models/setcover/model/model_best.ckpt \
    > logs/setcover/train-hard.log 2>&1 &

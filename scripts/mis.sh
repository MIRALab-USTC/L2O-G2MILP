mkdir -p logs/mis

# preprocess
nohup python preprocess.py dataset=mis num_workers=10 > logs/mis/preprocess.log 2>&1 &

# train
nohup python train.py dataset=mis cuda=0 num_workers=10 job_name=mis:default > logs/mis/train.log 2>&1 &

# generate
# ${dir} should be changed to your own path
nohup python generate.py dataset=mis generator.mask_ratio=0.01 cuda=0 num_workers=10 \
    dir=outputs/models/mis \
    > logs/mis/generate:0.01.log 2>&1 &
nohup python generate.py dataset=mis generator.mask_ratio=0.05 cuda=0 num_workers=10 \
    dir=outputs/models/mis \
    > logs/mis/generate:0.05.log 2>&1 &
nohup python generate.py dataset=mis generator.mask_ratio=0.1 cuda=0 num_workers=10 \
    dir=outputs/models/mis \
    > logs/mis/generate:0.1.log 2>&1 &

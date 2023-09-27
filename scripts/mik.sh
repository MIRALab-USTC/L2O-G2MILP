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


nohup python train.py dataset=mik cuda=1 num_workers=10 \
    trainer.beta.var.min=0.00001 trainer.beta.var.max=0.0001 \
    job_name=mik:beta_var:0.00001:0.0001 \
    > logs/mik/train1.log 2>&1 &

nohup python train.py dataset=mik cuda=2 num_workers=10 \
    trainer.beta.var.min=0.00001 trainer.beta.var.max=0.00005 \
    job_name=mik:beta_var:0.00001:0.00005 \
    > logs/mik/train2.log 2>&1 &

nohup python train.py dataset=mik cuda=4 num_workers=10 \
    trainer.beta.var.min=0.00001 trainer.beta.var.max=0.00005 \
    trainer.beta.var.anneal_period=20000 \
    job_name=mik:beta_var:0.00001:20k:0.00005 \
    > logs/mik/train4.log 2>&1 &

nohup python train.py dataset=mik cuda=5 num_workers=10 \
    trainer.beta.var.min=0.000001 trainer.beta.var.max=0.000005 \
    job_name=mik:beta_var:0.000001:0.000005 \
    > logs/mik/train5.log 2>&1 &

nohup python train.py dataset=mik cuda=6 num_workers=10 \
    trainer.beta.var.min=0.000001 trainer.beta.var.max=0.00001 \
    trainer.beta.var.anneal_period=20000 \
    job_name=mik:beta_var:0.000001:20k:0.00001 \
    > logs/mik/train6.log 2>&1 &

nohup python train.py dataset=mik cuda=7 num_workers=10 \
    trainer.beta.var.min=0.00001 trainer.beta.var.max=0.00005 \
    model.loss_weights.weights_loss=3.0 \
    job_name=mik:beta_var:0.00001:0.00005:weights:3.0 \
    > logs/mik/train7.log 2>&1 &
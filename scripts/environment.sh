conda create -n g2milp python=3.7
conda activate g2milp
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c pyg pyg
conda install -c conda-forge ecole
conda install networkx
conda install pandas
conda install -c conda-forge pyscipopt
pip install hydra-core --upgrade
pip install community
pip install python-louvain
pip install tensorboardX
pip install tensorboard
conda install ipykernel --update-deps --force-reinstall
python -m pip install gurobipy
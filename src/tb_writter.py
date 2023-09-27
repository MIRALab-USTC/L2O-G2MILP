from tensorboardX import SummaryWriter

tb_writter: SummaryWriter = None
step: int = 0

def set_logger(log_dir: str):
    global tb_writter
    global step
    tb_writter = SummaryWriter(log_dir)
    step = 0

def set_step(total_step: int):
    global step
    step = total_step

def add_scalar(tag, scalar_value, global_step=None, walltime=None):
    global tb_writter
    tb_writter.add_scalar(tag, scalar_value, global_step, walltime)
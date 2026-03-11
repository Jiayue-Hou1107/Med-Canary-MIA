import argparse
import os
import constants

def get_parser():
    ''' parse arguments '''

    parser = argparse.ArgumentParser()

    ### args agnostic to discrete or soft prompting
    parser.add_argument(
        "--model",
        help="model name"
    )
    parser.add_argument(
        "--case_id",
        type=int,
        help="case id number (integer) per constants.py"
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='id of gpu to use (for single GPU training only)'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default="0",
        help='id of gpu to use (for single GPU training only)'
    )
    
    # === 新增：断点续训参数 ===
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to resume training from (e.g., .../models_tuned/xxx/3)"
    )
    # ========================
    
    args = parser.parse_args()
    args = set_args(args)
    
    # For distributed training, device will be set in main()
    # For single GPU training, use the specified gpu_id
    if 'RANK' not in os.environ:
        args.device = f'cuda:{args.gpu_id}'

    return args


def set_args(args):
    ''' set args based on parser, constants.py 
        written separately to be modular w generate_table.py '''
    
    # define directories based on expmt params
    args.expmt_name = f'{args.model}_case{args.case_id}'
    args = set_args_dir_out(args)
    args = set_args_dir_model(args)

    args.max_new_tokens = constants.cases[args.case_id]['max_new_tokens']
    if args.case_id >= 100:
        args.batch_size = constants.cases[args.case_id]['batch_size']
        args.trn_epochs = constants.cases[args.case_id]['trn_epochs']
        args.grad_accum_steps = constants.cases[args.case_id]['grad_accum_steps']
        args.lr_n_warmup_steps = constants.cases[args.case_id]['lr_n_warmup_steps']

    return args


def set_args_dir_out(args):
    ''' create directory for output data '''

    args.dir_out = os.path.join(
        constants.DIR_PROJECT,
        'output',
        args.expmt_name + '/'
    )

    # Use exist_ok=True to avoid race condition in distributed training
    os.makedirs(args.dir_out, exist_ok=True)

    return args


def set_args_dir_model(args):
    ''' create directory for tuned models '''

    args.dir_models_tuned = os.path.join(
        constants.DIR_PROJECT,
        'models_tuned',
        args.expmt_name + '/'
    )
   
    # Use exist_ok=True to avoid race condition in distributed training
    os.makedirs(args.dir_models_tuned, exist_ok=True)

    return args
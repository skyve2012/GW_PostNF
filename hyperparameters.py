import argparse
import logging
logger = logging.getLogger(__name__)
def create_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parent parser for tape functions',
                                     add_help=False)
    
    parser.add_argument('--vocab_file', default=None,
                        help='Pretrained tokenizer vocab file')
    parser.add_argument('--output_dir', default='/home/hongyu2/debugs/', type=str)
    parser.add_argument('--device', default='cpu', type=str, help="used for specify device for datatset output")
    parser.add_argument('--data_path', default='/home/hongyu2/data/', type=str,
                        help='Directory from which to load task data')
    parser.add_argument('--no_cuda', action='store_true', help='CPU-only flag')
    parser.add_argument('--distributed_train', action='store_true', help='start distributed training')
    parser.add_argument('--seed', default=42, type=int, help='Random seed to use')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank of process in distributed training. '
                             'Set by launch script.')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to use for multi-threaded data loading')
    parser.add_argument('--log_level', default=logging.INFO,
                        choices=['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR',
                                 logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
                        help="log level for the experiment")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--num_log_iter', default=40, type=int,
                        help='Number of training steps per log iteration')
    parser.add_argument('--cuda_idx', default='1', type=str, help='cuda index for single GPU train')

    return parser


def create_train_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Training on the TAPE datasets',
                                     parents=[base_parser])
    parser.add_argument('--model_type', help='Base model class to run', default='NF_new', type=str)
    parser.add_argument('--model_config_file', default=None, type=dict,
                        help='Config file for model')
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='temperature for contrastive learning')
    parser.add_argument('--mode', default="train", type=str,
                        help='Run mode (trian, test)')
    parser.add_argument('--model_base', default="wavenet1", type=str,
                        help='the name of the major model architecture in a bigger structure')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--pyro', action='store_true', help='if use pyro NF')
    parser.add_argument('--simCLR', action='store_true', help='if add simCLR loss in the model or not')
    parser.add_argument('--low_res', action='store_true', help='if use 4k data or not')
    parser.add_argument('--short', action='store_true', help='if use 4k data or not')
    parser.add_argument('--scheduler', action='store_true', help='whether or not to activate scheduler')
    parser.add_argument('--lambda_gp', default=10., type=float,
                        help='peanlty for gradient GP')
    parser.add_argument('--n_parameter', default=2, type=int,
                        help='# of parameters for model')
    parser.add_argument('--n_flow', default=5, type=int,
                        help='# of normalizing flows for model')
    parser.add_argument('--n_repeat_phase2', default=3, type=int,
                        help='for model')
    parser.add_argument('--n_sample_phase2', default=5, type=int,
                        help='for model')
    parser.add_argument('--wave_num_blocks', default=10, type=int,
                        help='for Wavenet')
    parser.add_argument('--wave_num_layers', default=8, type=int,
                        help='for Wavenet')
    parser.add_argument('--wave_num_hidden', default=32, type=int,
                        help='for Wavenet')
    parser.add_argument('--wave_kernel_size', default=2, type=int,
                        help='for Wavenet')
    parser.add_argument('--filter_size', default=4, type=int,
                        help='filter_size for resnet')
    parser.add_argument('--input_size', default=8192, type=int,
                        help='input size of the signal')
    parser.add_argument('--output_dim', default=256, type=int,
                        help='its meaning can vary: e.g. for contrastive learning is the z dim, for PE, its the # of parameters')
    parser.add_argument('--n_critic', default=2, type=int,
                        help='for model generator update')
    parser.add_argument('--pretrained_model_dir', default='/home/hongyu2/debugs/', type=str)
    parser.add_argument('--pretrained_model_checkpoint', default=None, type=str)
    parser.add_argument('--online_whiten', action='store_true', help='whether or not whiten data online')
    parser.add_argument('--output_feature_dim', default=256, type=int,
                        help='its meaning can vary: e.g. for contrastive learning is the z dim, for PE, its the # of parameters')
    parser.add_argument('--pretrained', type=bool, help='load pre-trained model', default=False)
    parser.add_argument('--pool_only', type=bool, help='blurpool for invariant cnn', default=True)
    parser.add_argument('--use_cosine_similarity', type=bool, help='load pre-trained model', default=True) 
    parser.add_argument('--dataset_name', default='GW170104', type=str,
                        help='Directory from which to load task data')
    parser.add_argument('--num_train_epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16 weights')
    parser.add_argument('--warmup_steps', default=1000, type=int,
                        help='Number of learning rate warmup steps')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of forward passes to make for each backwards pass')
    parser.add_argument('--loss_scale', default=0, type=int,
                        help='Loss scaling. Only used during fp16 training.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Maximum gradient norm')
    parser.add_argument('--flooding_b', default=0., type=float,
                        help='b value for flooding loss (0. meaning there is no flooding)')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Name to give to this experiment')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help="Frequency of eval pass. A value <= 0 means the eval pass is "
                             "not run")
    parser.add_argument('--save_freq', default=1, type=int,
                        help="How often to save the model during training. Either an integer "
                             "frequency or the string 'improvement'")
    parser.add_argument('--patience', default=-1, type=int,
                        help="How many epochs without improvement to wait before ending "
                             "training")
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help="whether to resume training from the checkpoint")
    return parser
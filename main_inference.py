import os
import random
import argparse
import shutil
import numpy as np
import torch
import logging

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_file",
    #                     type=str,
    #                     default="",
    #                     help="data config file full path")
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        default="",
                        help="model config file")
    parser.add_argument("--out_db_file",
                        type=str,
                        help="If specified, will write result here.")
    parser.add_argument("opts",
                        default=None,
                        nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    parser.add_argument(
        "--logging_level",
        type=int,
        choices=[10, 20, 30, 40],
        default=20,
        help="Set logging level. 10: debug, 20: info, 30: warning, 40: error.")
    return parser


def main(args):
    cfg_data_file = os.path.join("./configs/data/stamps.yaml")  #args.data_file
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join(["dummy", args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)

    print("** Config **")
    print(cfg)
    print("************")

    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print("Model directory: {}".format(cfg.model_dir))

    if args.out_db_file is not None:
        shutil.copyfile(cfg.test_db_file, args.out_db_file)
        cfg.test_db_file = args.out_db_file

    cfg.test_only = True

    trainer = Trainer(cfg, is_inference=True)
    trainer.load_model(cfg.model_dir)
    trainer.inference(commit=(args.out_db_file is not None))


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(
        format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=args.logging_level)

    main(args)

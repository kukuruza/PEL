import os
import time
import datetime
import pprint
import logging
import progressbar
import simplejson as json
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224

from shuffler.interface.pytorch import datasets

from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(
        url, root="/ocean/projects/hum210002p/shared/classification/cache")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(cfg):
    backbone_name = cfg.backbone
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp16":
        # ViT's default precision is fp32
        model.half()

    return model


def load_decoding(encoding_file):
    # Read the encoding, and pprepare the decoding table.
    if not os.path.exists(encoding_file):
        raise FileNotFoundError('Cant find the encoding file: %s' %
                                encoding_file)
    with open(encoding_file) as f:
        encoding = json.load(f)
    decoding = {}
    for name, name_id in encoding.items():
        if name_id == -1:
            assert False
            decoding[name_id] = None
        elif name_id in decoding:
            raise ValueError('Not expecting multiple back mapping.')
        else:
            decoding[name_id] = name
    print('Have %d entries in decoding.' % len(decoding))
    return decoding


class Trainer:
    def __init__(self, cfg, is_inference=False):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg

        self.decoding = load_decoding(cfg.encoding_file)
        self.classnames = [value for _, value in self.decoding.items()]

        if is_inference:
            self.build_inference_data_loader()
        else:
            self.build_training_data_loader()
            self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs,
                                       self.few_idxs)

        self.build_model()
        self._writer = None

    def build_training_data_loader(self):
        cfg = self.cfg
        resolution = cfg.resolution

        if cfg.debug_no_albumentation:
            train_image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            albumentation_tranform = A.Compose([
                A.CLAHE(),
                A.ShiftScaleRotate(shift_limit=0.1,
                                   scale_limit=(0, 0.35),
                                   rotate_limit=20,
                                   p=1.,
                                   border_mode=cv2.BORDER_REPLICATE),
                A.Blur(blur_limit=1),
                A.OpticalDistortion(),
                A.GridDistortion(),
                A.Resize(224, 224),
                A.HueSaturationValue(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
            train_image_transform = lambda x: albumentation_tranform(image=x)[
                'image']
        test_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        used_keys = [
            'imagefile', 'image', 'objectid', 'name_id', 'name', 'x_on_page',
            'width_on_page', 'y_on_page', 'height_on_page'
        ]
        common_transform_group = {
            'name_id':
            lambda x: int(x),
            'x_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
            'y_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
            'width_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
            'height_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
        }
        train_transform_group = dict(common_transform_group)
        train_transform_group.update({'image': train_image_transform})
        test_transform_group = dict(common_transform_group)
        test_transform_group.update({'image': test_image_transform})

        # Limit the number of objects.
        object_id_sql_clause = (
            'objectid IN '
            '(SELECT objectid FROM properties WHERE key = "name_id" '
            'AND CAST(value AS INT) < "%d")' % cfg.debug_num_train_classes)
        names_and_count_sql = (
            'SELECT DISTINCT(name),COUNT(1) FROM objects '
            'JOIN properties ON objects.objectid = properties.objectid '
            'WHERE key = "name_id" AND CAST(value AS INT) < "%d"'
            'GROUP BY name ORDER BY CAST(value AS INT)' %
            cfg.debug_num_train_classes)

        train_dataset = datasets.ObjectDataset(
            cfg.train_db_file,
            rootdir=cfg.rootdir,
            where_object=object_id_sql_clause,
            mode='r',
            used_keys=used_keys,
            transform_group=train_transform_group)
        print("Total number of train samples:", len(train_dataset))

        train_init_dataset = datasets.ObjectDataset(
            cfg.train_db_file,
            rootdir=cfg.rootdir,
            where_object=object_id_sql_clause,
            mode='r',
            used_keys=used_keys,
            transform_group=train_transform_group)

        train_test_dataset = datasets.ObjectDataset(
            cfg.train_db_file,
            rootdir=cfg.rootdir,
            where_object=object_id_sql_clause,
            mode='r',
            used_keys=used_keys,
            transform_group=test_transform_group)

        # Set num_classes everywhere.
        names_and_count = train_dataset.execute(names_and_count_sql)
        classnames, cls_num_list = zip(*names_and_count)
        # pprint.pprint(names_and_count)
        num_classes = len(classnames)
        print('num_classes:', num_classes)

        # Test dataset needs to have classes present in training dataset.
        names_str = "'" + "', '".join(classnames) + "'"
        # print('Querying the test database for:', names_str)
        test_dataset = datasets.ObjectDataset(
            cfg.test_db_file,
            rootdir=cfg.rootdir,
            where_object="name IN (%s)" % names_str,
            mode='r',
            used_keys=used_keys,
            transform_group=test_transform_group)
        print("Total number of test samples:", len(test_dataset))

        self.num_classes = num_classes
        self.cls_num_list = cls_num_list
        self.classnames = classnames

        split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) &
                         (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=cfg.micro_batch_size,
                                       shuffle=True,
                                       num_workers=cfg.num_workers,
                                       pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
                                            batch_size=64,
                                            sampler=init_sampler,
                                            shuffle=False,
                                            num_workers=cfg.num_workers,
                                            pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
                                            batch_size=64,
                                            shuffle=False,
                                            num_workers=cfg.num_workers,
                                            pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
                                      batch_size=64,
                                      shuffle=False,
                                      num_workers=cfg.num_workers,
                                      pin_memory=True)

        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_inference_data_loader(self):
        cfg = self.cfg
        resolution = cfg.resolution

        test_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        used_keys = [
            'imagefile', 'image', 'objectid', 'name_id', 'name', 'x_on_page',
            'width_on_page', 'y_on_page', 'height_on_page'
        ]
        common_transform_group = {
            'name_id':
            lambda x: int(x) if x is not None and x != 'None' else -1,
            'x_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
            'y_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
            'width_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
            'height_on_page':
            lambda x: float(x) if x is not None and x != 'None' else -1,
        }
        test_transform_group = dict(common_transform_group)
        test_transform_group.update({'image': test_image_transform})

        # with open(args.encoding_file, 'r') as f:
        #     encoding = json.loads(f.read())
        # model.name_encoding = encoding

        # Test dataset needs to have classes present in training dataset.
        self.inference_dataset = datasets.ObjectDataset(
            cfg.test_db_file,
            rootdir=cfg.rootdir,
            mode='w',
            used_keys=used_keys,
            transform_group=test_transform_group,
            copy_to_memory=False)
        print("Total number of test samples:", len(self.inference_dataset))

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            prompts = self.get_tokenized_prompts(classnames)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in [
                    "class_mean", "1_shot", "10_shot", "100_shot"
            ]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")

            torch.cuda.empty_cache()

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), use all of them!"
            )
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{
            "params": self.tuner.parameters()
        }, {
            "params": self.head.parameters()
        }],
                                     lr=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal":  # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM":  # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB":  # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW":  # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS":  # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA":  # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE":  # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        prompts = self.get_tokenized_prompts(classnames)
        text_features = self.model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels,
                                                   return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx:idx + cnt].mean(dim=0,
                                                              keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs",
                                 max_iter=100,
                                 penalty="l2",
                                 class_weight="balanced").fit(
                                     all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(
            self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch['image']
                label = batch['name_id']
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec == "amp":
                    with autocast():
                        output = self.model(image)
                        loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step
                            == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step
                            == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (num_epochs - epoch_idx - 1) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [
                        f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"
                    ]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [
                        f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"
                    ]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [
                        f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"
                    ]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val,
                                        n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg,
                                        n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)

                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()

            self.test()
            torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch['image']
            label = batch['name_id']

            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate(self.num_classes)

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    @torch.no_grad()
    def inference(self, commit):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()

        for sample in progressbar.progressbar(self.inference_dataset):
            image = sample['image']
            images = torch.unsqueeze(image, 0).to(self.device)
            outputs = self.model(images)
            assert len(outputs) == 1

            pred = outputs.max(1)[1]
            conf = torch.softmax(outputs, dim=1).max(1)[0]

            pred = pred.data.cpu().numpy().tolist()
            conf = conf.data.cpu().numpy().tolist()

            objectid = int(sample['objectid'])
            name_id = int(pred[0])
            score = float(conf[0])
            if name_id not in self.decoding:
                raise ValueError('name_id %d not in decoding.')
            name = self.decoding[name_id]
            logging.debug(
                'Setting name_id %s (name_id %d) with score %.3f to object %d',
                name, name_id, score, objectid)
            self.inference_dataset.execute(
                'UPDATE objects SET name=?,score=? WHERE objectid=?',
                (name, score, objectid))

        if commit:
            self.inference_dataset.conn.commit()
        self.inference_dataset.close()

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {"tuner": tuner_dict, "head": head_dict}

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                'Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict)
        self.head.load_state_dict(head_dict)

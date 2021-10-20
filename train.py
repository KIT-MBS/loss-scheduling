#!/usr/bin/env python3
"""
adapted from github.com/pytorch/vision/blob/master/references/segmentation/train.py
"""

import os
import time
import datetime
import math
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.nn import CrossEntropyLoss
from loss import DiceLoss, FocalLoss, MaxPooledLoss, CompoundLoss

import torchvision

from tqdm import tqdm


from calibration import CalibrationHistogram
from unet import unet_resnet50
import transforms as T


def get_transforms(train, base_size=520, crop_size=480):

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def get_dataset(name, image_set, transforms):
    from coco_utils import get_coco
    data_root_dir = os.environ["DATA_PATH"]

    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)

    def cityscapes(root, image_set='train', transforms=None):
        return torchvision.datasets.Cityscapes(root, split=image_set, mode='fine', target_type='semantic', transforms=transforms)
    datasets = {
        'sbd': (data_root_dir+'sbd/', sbd, 21),
        'coco': (data_root_dir+'coco/', get_coco, 21),
        'voc': (data_root_dir+'PascalVOC2012/', torchvision.datasets.VOCSegmentation, 21),
        'cityscapes': (data_root_dir+'cityscapes/', cityscapes, 34)
        }

    root, fn, num_classes = datasets[name]

    ds = fn(root, image_set=image_set, transforms=transforms)
    return ds, num_classes


def get_criterion(loss, secondary_loss, loss_weights, secondary_loss_weights, loss_scheduler):
    loss_dict = {'ce': CrossEntropyLoss, 'dl': DiceLoss, 'fl': FocalLoss, 'mp': MaxPooledLoss}
    # loss_scheduler_dict = {'' : None, 'ramp' : RampLossWeightScheduler, 'decay' : DecayLossWeightScheduler}

    if loss_weights or secondary_loss_weights:
        import pickle
        with open('class_freq.pkl', 'rb') as f:
            class_weight = pickle.load(f)[dataset]['train']
        class_weight = 1. / class_weight
        class_weight /= class_weight.mean()
        class_weight = class_weight.to(dtype=torch.float32)

    loss_weights = class_weight if loss_weights else None
    secondary_loss_weights = class_weight if secondary_loss_weights else None

    if secondary_loss != '':
        secondary_loss = loss_dict[secondary_loss]
        if loss == 'mp':
            loss = MaxPooledLoss(secondary_loss, ignore_index=255)
        else:
            loss = loss_dict[loss]
            loss = CompoundLoss(secondary_loss, loss, kwargs1={'weight': secondary_loss_weights}, kwargs2={'weight': loss_weights}, weight=[.5, .5], ignore_index=255)
    else:
        loss = loss_dict[loss](ignore_index=255)

    return loss


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()

        correct = torch.diag(h)
        positive = h.sum(1)
        predicted_positive = h.sum(0)

        acc_global = correct.sum() / h.sum()
        acc = correct[positive > 0] / positive[positive > 0]  # NOTE this is per class recall
        iu = correct[positive > 0] / (positive[positive > 0] + predicted_positive[positive > 0] - correct[positive > 0])
        return acc_global, acc, iu

    def precision(self):
        h = self.mat.float()
        positive = h.sum(1)
        return torch.diag(h)[positive > 0] / h.sum(0)[positive > 0]

    def recall(self):
        h = self.mat.float()
        positive = h.sum(1)
        return torch.diag(h)[positive > 0] / h.sum(1)[positive > 0]

    def macro_f1(self):
        p = self.precision()
        r = self.recall()
        return (2*(p * r)/(p + r)).mean()

    def macro_f(self, beta=1):
        p = self.precision()
        r = self.recall()
        return ((1+beta**2) * (p * r)/(p*beta**2 + r)).mean()

    def macro_f2(self):
        return self.macro_f(beta=2)

    def macro_fpoint5(self):
        return self.macro_f(beta=0.5)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


# NOTE do these depending on dataset size, at least one epoch winddown, several epochs warmup
class RampLossWeightScheduler():
    def __init__(self, loss_obj, transition_start, transition_finish, t=None):
        self._step_count = 0
        self.loss = loss_obj
        if t is None:
            self.t = loss_obj.w[1]
        else:
            self.t = t
            self.loss.reweight([1.-self.t, self.t])
        self.start_t = self.t

        self.transition_start = transition_start
        self.transition_finish = transition_finish
        self.factor = (1. - self.start_t) / (self.transition_finish - self.transition_start)
        return

    def step(self):
        self._step_count += 1
        self.t = max(self.start_t, min(1, self.start_t - self.transition_start*self.factor + self.factor*self._step_count))

        self.loss.reweight([1.-self.t, self.t])
        return


class DecayLossWeightScheduler():
    def __init__(self, loss_obj, sinus_start, sinus_end, t=None):
        self._step_count = 0
        self.loss = loss_obj
        if t is None:
            self.t = loss_obj.w[1]
        else:
            self.t = t
            self.loss.reweight([1.-self.t, self.t])
        self.start_t = self.t

        self.sinus_start = sinus_start
        self.sinus_end = sinus_end
        self.factor = self.start_t / self.sinus_start
        return

    def step(self):
        self._step_count += 1
        self.t = (min(self.start_t, max(0, self.start_t - self.factor*self._step_count)) +
                  1 - math.cos(min(math.pi/2, max(0, (self._step_count - self.sinus_start) * math.pi/2/(self.sinus_end - self.sinus_start)))))

        self.loss.reweight([1.-self.t, self.t])
        return


def evaluate(model, data_loader, device, num_classes, criterion=None, metric_writer=None, epoch=None, disable_progress=True):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    calhist = CalibrationHistogram(10, ignore_index=255)

    losses = []

    with torch.no_grad():
        i = 0
        running_ce = 0.
        for image, target in tqdm(data_loader, disable=disable_progress):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            loss = criterion(output, target)
            cross_entropy = torch.nn.functional.cross_entropy(output, target, ignore_index=255)
            if metric_writer is not None:
                losses.append(loss)
                if i < 5:
                    metric_writer.add_image('sample{}'.format(i), image.squeeze())
                    metric_writer.add_image('target{}'.format(i), target.squeeze(), dataformats='HW')
                    metric_writer.add_image('prediction{}'.format(i), (output.argmax(1)).squeeze(), dataformats='HW')

            confmat.update(target.flatten(), output.argmax(1).flatten())

            calhist.update(torch.nn.functional.softmax(output, dim=1), target)

            running_ce += cross_entropy
            i += 1
    if metric_writer is not None:
        glacc, acc, iu = confmat.compute()
        meaniu = iu.mean()

        metric_writer.add_scalar('val_loss', sum(losses)/len(losses), epoch)
        metric_writer.add_scalar('val_ce', running_ce/len(data_loader), epoch)
        metric_writer.add_scalar('acc', glacc, epoch)
        metric_writer.add_scalar('mean_IoU', meaniu, epoch)
        metric_writer.add_scalar('balanced_acc', acc.mean(), epoch)
        metric_writer.add_scalar('calibration_score', calhist.score(), epoch)

        confusion = confmat.mat.clone().detach()
        per_class = confusion.sum(dim=0)
        confusion /= per_class.unsqueeze(0).expand(confusion.size())
        metric_writer.add_image('confusion', confusion, epoch, dataformats='HW')

    return confmat, calhist, running_ce/len(data_loader)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, lr_scheduler=None, criterion_scheduler=None, metric_writer=None, disable_progress=True):
    model.train()

    i = epoch * len(data_loader)
    for image, target in tqdm(data_loader, disable=disable_progress):
        image, target = image.to(device), target.to(device)
        output = model(image)['out']  # NOTE ignoring the aux_loss
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cross_entropy = torch.nn.functional.cross_entropy(output, target, ignore_index=255)
        if metric_writer is not None:
            metric_writer.add_scalar('ce', cross_entropy, i)
            metric_writer.add_scalar('loss', loss, i)
            metric_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
            if criterion_scheduler is not None:
                metric_writer.add_scalar('ce_weight', criterion_scheduler.t, i)

        if lr_scheduler is not None:
            lr_scheduler.step()
        if criterion_scheduler is not None:
            criterion_scheduler.step()

        i += 1
    return


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def train(
        output_dir,
        criterion,
        criterion_scheduler='',
        epochs=30,
        batch_size=12,
        device=None,
        num_workers=16,
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        dataset='voc',
        model='fcn_resnet101',
        train_writer=None,
        val_writer=None,
        pretrained=False,
        disable_progress=True):

    os.makedirs(output_dir, exist_ok=True)

    print(device, ": getting data...")
    if dataset != 'cityscapes':
        trainset, num_classes = get_dataset(dataset, "train", get_transforms(train=True))
        valset, _ = get_dataset(dataset, "val", get_transforms(train=False))
    else:
        trainset, num_classes = get_dataset(dataset, "train", get_transforms(train=True, base_size=820, crop_size=780))
        valset, _ = get_dataset(dataset, "val", get_transforms(train=False))

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn)

    print(device, ": samples: ", len(trainset))
    print(device, ": batches: ", len(train_loader))
    print(device, ": val samples: ", len(val_loader))

    print(device, ": setting up model...")

    # NOTE aux_loss is True if pretrained
    if dataset == 'cityscapes':
        assert not pretrained

    model_name = model
    if model == 'unet_resnet50':
        assert not pretrained
        model = unet_resnet50(pretrained=pretrained, num_classes=num_classes, pretrained_encoder=True)
    else:
        model = torchvision.models.segmentation.__dict__[model](pretrained=pretrained, num_classes=num_classes, aux_loss=False, pretrained_backbone=True)

    criterion.to(device)
    model.to(device)

    if model_name == 'unet_resnet50':
        params_to_optimize = [
                {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
                {"params": [p for p in model.decoder.parameters() if p.requires_grad]},
                ]
    else:
        params_to_optimize = [{"params": [p for p in model.classifier.parameters() if p.requires_grad]}, ]

    optimizer = torch.optim.SGD(params_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(train_loader) * epochs)) ** 0.9)
    if criterion_scheduler == 'ramp':
        criterion_scheduler = RampLossWeightScheduler(criterion, transition_start=(0.3*len(train_loader)*epochs), transition_finish=0.9*len(train_loader)*epochs, t=0.)
    elif criterion_scheduler == 'decay':
        sinus_start = 0.5*len(train_loader)*epochs
        sinus_end = 0.9*len(train_loader)*epochs

        criterion_scheduler = DecayLossWeightScheduler(criterion, sinus_start, sinus_end, t=1.)
    elif criterion_scheduler == '':
        criterion_scheduler = None
    else:
        raise

    start_time = time.time()

    best_IoU = 0.
    best_cross_entropy = 1000000.
    best_accuracy = 0.
    best_bacc = 0.
    best_fscore = 0.

    best_score = 0.
    best_balanced_score = 0.
    print(device, ": training...")
    for epoch in range(epochs):
        train_one_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                device,
                epoch,
                lr_scheduler=lr_scheduler,
                criterion_scheduler=criterion_scheduler,
                metric_writer=train_writer,
                disable_progress=disable_progress)
        confmat, calhist, cross_entropy = evaluate(
                model,
                val_loader,
                device=device,
                num_classes=num_classes,
                criterion=criterion,
                metric_writer=val_writer,
                epoch=epoch,
                disable_progress=disable_progress)
        print(device, ': ', confmat)

        acc, IoU, bacc = confmat.compute()
        IoU = IoU.mean().item()
        acc = acc.item()
        bacc = bacc.mean().item()
        fscore = confmat.macro_f1()

        calibration_score = calhist.score()

        score = 2 * (calibration_score * acc) / (calibration_score + acc)
        balanced_score = 2*(calibration_score * fscore) / (calibration_score + fscore)

        torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'b': batch_size,
                    'lr': lr,
                    'momentum': momentum,
                    'weight_decay': weight_decay,
                    'cross_entropy': cross_entropy,
                    'confmat': confmat,
                    'calhist': calhist
                },
                os.path.join(output_dir, 'last_model.pth')
                )

        if best_IoU < IoU:
            best_IoU = IoU
            torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'b': batch_size,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'cross_entropy': cross_entropy,
                        'confmat': confmat,
                        'calhist': calhist
                    },
                    os.path.join(output_dir, 'best_iu_model.pth')
                    )

        if best_score < score:
            best_score = score
            torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'b': batch_size,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'cross_entropy': cross_entropy,
                        'confmat': confmat,
                        'calhist': calhist
                    },
                    os.path.join(output_dir, 'best_score_model.pth')
                    )
        if best_balanced_score < balanced_score:
            best_balanced_score = balanced_score
            torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'b': batch_size,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'cross_entropy': cross_entropy,
                        'confmat': confmat,
                        'calhist': calhist
                    },
                    os.path.join(output_dir, 'best_balanced_score_model.pth')
                    )

        if best_cross_entropy > cross_entropy:
            best_cross_entropy = cross_entropy
            torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'b': batch_size,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'cross_entropy': cross_entropy,
                        'confmat': confmat,
                        'calhist': calhist
                    },
                    os.path.join(output_dir, 'best_ce_model.pth')
                    )
        if best_accuracy < acc:
            best_accuracy = acc
            torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'b': batch_size,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'cross_entropy': cross_entropy,
                        'confmat': confmat,
                        'calhist': calhist
                    },
                    os.path.join(output_dir, 'best_acc_model.pth')
                    )
        if best_bacc < bacc:
            best_bacc = bacc
            torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'b': batch_size,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'cross_entropy': cross_entropy,
                        'confmat': confmat,
                        'calhist': calhist
                    },
                    os.path.join(output_dir, 'best_bacc_model.pth')
                    )
        if best_fscore < fscore:
            best_fscore = fscore
            torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'b': batch_size,
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'cross_entropy': cross_entropy,
                        'confmat': confmat,
                        'calhist': calhist
                    },
                    os.path.join(output_dir, 'best_f1_model.pth')
                    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training for Calibrated Segmentation Models')

    parser.add_argument('--dataset', default='voc', help='dataset')  # voc, coco, cityscapes
    parser.add_argument('--model', default='fcn_resnet101', help='model')  # fcn_resnet101, deeplabv3_resnet101, unet_resnet50
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--output-dir', default='', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--seed', default='')

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # loss parameters
    parser.add_argument('--loss-weights', action='store_true')
    parser.add_argument('--loss', default='ce')
    parser.add_argument('--secondary-loss', default='')
    parser.add_argument('--secondary-loss-weights', action='store_true')
    parser.add_argument('--loss-scheduler', default='')  # empty or ramp or decay
    parser.add_argument('--enable-progress', action='store_true')

    args = parser.parse_args()

    print(args)

    if args.seed != '':
        torch.manual_seed(int(args.seed))

    device = torch.device(args.device)
    dataset = args.dataset
    model = args.model
    aux_loss = args.aux_loss
    batch_size = args.batch_size
    epochs = args.epochs
    num_workers = args.workers
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    output_dir = args.output_dir
    resume = args.resume
    pretrained = args.pretrained

    disable_progress = not args.enable_progress

    loss = args.loss
    secondary_loss = args.secondary_loss
    loss_weights = args.loss_weights
    secondary_loss_weights = args.secondary_loss_weights
    loss_scheduler = args.loss_scheduler

    criterion = get_criterion(loss, secondary_loss, loss_weights, secondary_loss_weights, loss_scheduler)

    tag = loss
    if loss_weights:
        tag = 'w'+tag
    if loss_scheduler != '':
        tag = '_'+loss_scheduler+'2_'+tag
    tag = secondary_loss + tag
    if secondary_loss_weights:
        tag = 'w' + tag
    if pretrained:
        tag = 'prefine'
    tag = model+'_'+dataset+'_'+tag
    if output_dir == '':
        output_dir = './out/'+tag+'/'

    train_writer = SummaryWriter(comment=tag, filename_suffix=tag)
    val_writer = SummaryWriter(comment=tag+'_val', filename_suffix=tag+'_val')

    train(
            output_dir,
            criterion,
            criterion_scheduler=loss_scheduler,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dataset=dataset,
            model=model,
            train_writer=train_writer,
            val_writer=val_writer,
            pretrained=pretrained,
            disable_progress=disable_progress)

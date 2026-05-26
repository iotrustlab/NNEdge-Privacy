from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import accuracy, adjust_learning_rate
from shared_files import data_pre as data
from models.single_modality import acc_encoder, gyro_encoder


class MyUTDmodel_IMU(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.acc_encoder = acc_encoder(input_size)
        self.gyro_encoder = gyro_encoder(input_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(3840, 1280),
            torch.nn.BatchNorm1d(1280),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1280, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x_acc, x_gyro):
        acc_output = self.acc_encoder(x_acc)
        gyro_output = self.gyro_encoder(x_gyro)
        fused_feature = torch.cat((acc_output, gyro_output), dim=1)
        return self.classifier(fused_feature)


def parse_option():
    parser = argparse.ArgumentParser("UTD G4 IMU-only fine-tuning")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="classifier learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--lr_decay_epochs", type=str, default="350,400,450",
                        help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="learning rate decay rate")
    parser.add_argument("--dataset", type=str, default="train_C/label_216/", help="dataset")
    parser.add_argument("--num_class", type=int, default=27, help="number of classes")
    parser.add_argument("--num_of_trial", type=int, default=5, help="number of seeds")
    parser.add_argument("--condition_name", type=str, required=True, help="condition name for result folders")
    parser.add_argument("--result_root", type=str, default="./save_g4_imu", help="base folder for outputs")
    parser.add_argument("--acc_ckpt", type=str, default="", help="checkpoint providing acc_encoder weights")
    parser.add_argument("--gyro_ckpt", type=str, default="", help="checkpoint providing gyro_encoder weights")
    parser.add_argument("--acc_layer_key", type=str, default="acc_encoder.", help="checkpoint key prefix for acc")
    parser.add_argument("--gyro_layer_key", type=str, default="gyro_encoder.", help="checkpoint key prefix for gyro")
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-4, help="encoder learning rate")
    parser.add_argument("--cosine", action="store_true", help="use cosine annealing")
    parser.add_argument("--warm", action="store_true", help="warm-up for large-batch training")

    opt = parser.parse_args()
    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = [int(it) for it in iterations]
    return opt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_folder(opt, trial_id):
    dataset_token = opt.dataset.strip("/").replace("/", "_")
    opt.result_path = os.path.join(opt.result_root, opt.condition_name, dataset_token, f"trial_{trial_id}", "results")
    os.makedirs(opt.result_path, exist_ok=True)


def set_loader(opt):
    print("train IMU data:")
    x_train_acc, x_train_gyro, y_train = data.load_data_IMU(opt.dataset)
    print("test IMU data:")
    x_test_acc, x_test_gyro, y_test = data.load_data_IMU("test")

    train_dataset = data.Multimodal_dataset(x_train_acc, x_train_gyro, y_train)
    test_dataset = data.Multimodal_dataset(x_test_acc, x_test_gyro, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return train_loader, test_loader


def load_submodule_state(ckpt_path, layer_key):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if layer_key in key:
            key = key.replace(layer_key, "")
            key = key.replace("module.", "")
            new_state_dict[key] = value
    return new_state_dict


def set_model(opt):
    model = MyUTDmodel_IMU(input_size=1, num_classes=opt.num_class)
    criterion = torch.nn.CrossEntropyLoss()

    if opt.acc_ckpt:
        model.acc_encoder.load_state_dict(load_submodule_state(opt.acc_ckpt, opt.acc_layer_key))
    if opt.gyro_ckpt:
        model.gyro_encoder.load_state_dict(load_submodule_state(opt.gyro_ckpt, opt.gyro_layer_key))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for idx, (input_acc, input_gyro, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            input_acc = input_acc.cuda()
            input_gyro = input_gyro.cuda()
            labels = labels.cuda()
        bsz = input_acc.shape[0]

        output = model(input_acc, input_gyro)
        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\tBT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\tloss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1
                )
            )
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    confusion = np.zeros((opt.num_class, opt.num_class))
    label_list = []
    pred_list = []

    with torch.no_grad():
        end = time.time()
        for idx, (input_acc, input_gyro, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                input_acc = input_acc.cuda()
                input_gyro = input_gyro.cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            output = model(input_acc, input_gyro)
            loss = criterion(output, labels)

            losses.update(loss.item(), bsz)
            acc1, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()
            label_list.extend(rows)
            pred_list.extend(cols)
            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\tAcc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                    )
                )

    f1_test = f1_score(label_list, pred_list, average="macro")
    print(" * Acc@1 {top1.avg:.3f}\tF1-score {f1_test:.3f}\t".format(top1=top1, f1_test=f1_test))
    return losses.avg, top1.avg, confusion, f1_test, label_list, pred_list


def main():
    opt = parse_option()
    seeds = [42, 43, 44, 45, 46]

    for trial_id in range(opt.num_of_trial):
        opt = parse_option()
        set_folder(opt, trial_id)
        set_seed(seeds[trial_id])

        train_loader, test_loader = set_loader(opt)
        model, criterion = set_model(opt)

        optimizer = optim.Adam(
            [
                {"params": model.acc_encoder.parameters(), "lr": opt.encoder_learning_rate},
                {"params": model.gyro_encoder.parameters(), "lr": opt.encoder_learning_rate},
                {"params": model.classifier.parameters(), "lr": opt.learning_rate},
            ],
            weight_decay=opt.weight_decay,
        )

        record_acc = np.zeros(opt.epochs)
        record_f1 = np.zeros(opt.epochs)
        record_loss = np.zeros(opt.epochs)
        record_acc_train = np.zeros(opt.epochs)
        best_acc = 0
        best_f1 = 0

        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)
            time1 = time.time()
            loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
            time2 = time.time()
            print("Train epoch {}, total time {:.2f}, accuracy:{:.2f}".format(epoch, time2 - time1, train_acc))

            val_loss, val_acc, confusion, val_f1, label_list, pred_list = validate(test_loader, model, criterion, opt)
            if val_acc > best_acc:
                best_acc = val_acc
                best_f1 = val_f1

            record_loss[epoch - 1] = loss
            record_acc[epoch - 1] = val_acc
            record_f1[epoch - 1] = val_f1
            record_acc_train[epoch - 1] = train_acc

            np.savetxt(os.path.join(opt.result_path, "confusion.txt"), confusion)
            np.savetxt(os.path.join(opt.result_path, "label.txt"), np.array(label_list))
            np.savetxt(os.path.join(opt.result_path, "pred.txt"), np.array(pred_list))
            np.savetxt(os.path.join(opt.result_path, "loss.txt"), record_loss)
            np.savetxt(os.path.join(opt.result_path, "test_accuracy.txt"), record_acc)
            np.savetxt(os.path.join(opt.result_path, "test_f1.txt"), record_f1)
            np.savetxt(os.path.join(opt.result_path, "train_accuracy.txt"), record_acc_train)

        print("Trial: ", trial_id)
        print("best accuracy: {:.3f}".format(best_acc))
        print("best F1:{:.3f}".format(best_f1))
        print("last accuracy: {:.3f}".format(val_acc))
        print("final F1:{:.3f}".format(val_f1))


if __name__ == "__main__":
    main()

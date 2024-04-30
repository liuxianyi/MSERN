# coding = utf-8
from functools import partial
import pathlib
import random
from tensorboardX import SummaryWriter
import os
import time
import shutil
from easydict import EasyDict
from torch.utils.data import DataLoader
from sklearn import metrics
import yaml
from model import *
from utils import *
import logging

from dataset.LoadData import my_data_set



def main():
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    

    # load config
    with open(r"./args.yaml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt = EasyDict(opt['common'])

    # create save dir
    pathlib.Path(opt.logger_name).mkdir(parents=True, exist_ok=True)
    pathlib.Path(opt.curve_tensorb).mkdir(parents=True, exist_ok=True)
    pathlib.Path("/".join(opt.log_dir.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    # logger...
    logging.basicConfig(
        level=logging.INFO, 
        filename=opt.log_dir,
        filemode='a', 
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

    writer = SummaryWriter(log_dir=opt.curve_tensorb, flush_secs=5)

    # train, val, test, split
    train_set = my_data_set(opt, train=True)
    test_set = my_data_set(opt, train=False)
    batch_size = opt.BatchSize  #32
    
    ## Load data loaders
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=True)

    # load model
    model = MMFSTSR(opt)

    best_accuracy = 0 

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}' (epoch {}, best_accuracy {})".
                  format(opt.resume, start_epoch, best_accuracy))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    for epoch in range(opt.epochs):
        model.scheduler.step()
        writer.add_scalar(
            'learning rate on net',
            model.optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, writer)

        # evaluate on validation set
        mAP = validate(opt, test_loader, model, epoch, writer)  
        # model.scheduler.step(mAP)

        # remember best accuracy and save checkpoint
        is_best = mAP > best_accuracy  
        best_accuracy = max(mAP, best_accuracy) 
        if is_best:
            save_checkpoint({  
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_accuracy': best_accuracy,
                'opt': opt,
            }, is_best, filename='checkpoint_' + str(epoch) + '.pth.tar', prefix=opt.logger_name + '/')

    print(' *** best={best:.4f}\t'.format(best=best_accuracy))


def train(opt, train_loader, model, epoch, writer):
    print("start to train")

    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train_start()
    since = time.time()
    print("start loading data ...")
    print("learning rate:",
          model.optimizer.state_dict()['param_groups'][0]['lr'])

    loss_epoch = 0
    map = 0
    ap = 0
    recall = 0

    for i, train_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - since) 

        # Update the model
        output, loss_batch = model.train_emb(
            *train_data)  

        # y_pred = torch.sigmoid(y_hat)
        loss_epoch = loss_batch + loss_epoch

        # measure elapsed time
        batch_time.update(time.time() - since)  

        # metrics
        map_batch = cal_ap(output, train_data[3]).mean()
        map = map + map_batch

        preds = torch.max(output, 1)[1]
        ap_batch = metrics.precision_score(train_data[4],
                                           preds.cpu(),
                                           average='weighted',
                                           zero_division=1)
        ap = ap + ap_batch

        recall_batch = metrics.recall_score(train_data[4],
                                            preds.cpu(),
                                            average='weighted',
                                            zero_division=1)
        recall = recall + recall_batch

        if i % 5 == 0:
            print(
                'Train: [{0}/{1}]\t'
                'batch_time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'data_time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss_batch {loss_batch:.4f}\t'
                'map_batch {map_batch:.4f}\t'
                'ap_batch {ap_batch:.4f}\t'
                'recall_batch {recall_batch:.4f}\t'
                'learning_rate {lr:.7f}'.format(
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss_batch=loss_batch,
                    map_batch=map_batch,
                    ap_batch=ap_batch,
                    recall_batch=recall_batch,
                    lr=model.optimizer.state_dict()['param_groups'][0]['lr']))

    # avg
    map = map / len(train_loader)
    ap = ap / len(train_loader)
    recall = recall / len(train_loader)
    print('Epoch: [{0}]\t'
          'Train Loss_epoch {loss:.4f} \t'
          'Loss_avg {loss_avg:.4f} \t'
          'mAP {map:.4f}\t'
          'ap {ap:.4f}\t'
          'recall {recall:.4f}\t'
          'learning_rate {lr:.7f}'.format(
              epoch,
              loss=loss_epoch,
              loss_avg=loss_epoch / len(train_loader),
              map=map,
              ap=ap,
              recall=recall,
              lr=model.optimizer.state_dict()['param_groups'][0]['lr']))

    # # Record logs in tensorboard
    if epoch % opt.log_step == 0:
        logging.info(
            'Train: Epoch: [{0}]\t'
            'Loss_avg {loss_avg:.4f} \t'
            'mAP {map:.4f}\t'
            'ap {ap:.4f}\t'
            'recall {recall:.4f}\t'
            'learning_rate {lr:.7f}'.format(
                epoch,
                loss_avg=loss_epoch / len(train_loader),
                map=map,
                ap=ap,
                recall=recall,
                lr=model.optimizer.state_dict()['param_groups'][0]['lr']))
    writer.add_scalar('Train_Loss', loss_epoch / len(train_loader), epoch)
    writer.add_scalar('Train_mAP', map, epoch)
    writer.add_scalar('Train_AP', ap, epoch)
    writer.add_scalar('Train_Recall', recall, epoch)


def validate(opt, test_loader, model, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_logger = LogCollector()
    end = time.time()
    running_loss = 0
    map = 0
    ap = 0
    recall = 0
    with torch.no_grad():
        print("start validate")
        print("start loading val data...")
        model.eval_start()

        for i, test_data in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)  

            # make sure val logger is used
            model.logger = val_logger

            # compute the output and loss
            y_hat, loss, private, public = model.test_emb(*test_data)
            if opt.visual_ts and epoch % 2 == 0:
                private_vis = f"./visual_ts/private/{epoch}"
                public_vis = f"./visual_ts/public/{epoch}"
                if not os.path.exists(private_vis):
                    os.makedirs(private_vis)
                if not os.path.exists(public_vis):
                    os.makedirs(public_vis)
                pri = {"data": private.cpu().data.numpy(), "label": test_data[4].data.numpy()}
                pub = {"data": public.cpu().data.numpy(), "label": test_data[4].data.numpy()}
                torch.save(pri, f"./visual_ts/private/{epoch}/{i}.pt")
                torch.save(pub, f"./visual_ts/public/{epoch}/{i}.pt")


            running_loss = loss + running_loss

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            map_batch = cal_ap(y_hat, test_data[3]).mean()
            map = map + map_batch

            preds = torch.max(y_hat, 1)[1]

            ap_batch = metrics.precision_score(test_data[4],
                                               preds.cpu(),
                                               average='weighted',
                                               zero_division=1)
            ap = ap + ap_batch

            recall_batch = metrics.recall_score(test_data[4],
                                                preds.cpu(),
                                                average='weighted',
                                                zero_division=1)
            recall = recall + recall_batch

            # Record logs in tensorboard
            print('Test: [{0}/{1}]\t'
                  'batch_time {batch_time.val:.3f} (batch_time.avg:.3f)\t'
                  'data_time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss:.4f}\t'
                  'mAP_batch {map_batch: .4f}\t'
                  'ap_batch {ap_batch: .4f}\t'
                  'recall_batch {recall_batch:.4f}'.format(
                      i,
                      len(test_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=loss,
                      map_batch=map_batch,
                      ap_batch=ap_batch,
                      recall_batch=recall_batch))

    map = map / len(test_loader)
    ap = ap / len(test_loader)
    recall = recall / len(test_loader)
    print('Epoch: [{0}]\t'
          'Test Loss_epoch {loss:.4f} \t'
          'Loss_avg {loss_avg:.4f} \t'
          'mAP {map:.4f}\t'
          'ap {ap: .4f}\t'
          'recall {recall:.4f}'.format(epoch,
                                       loss=running_loss,
                                       loss_avg=running_loss /
                                       len(test_loader),
                                       map=map,
                                       ap=ap,
                                       recall=recall))

    writer.add_scalar('Test_Loss', running_loss / len(test_loader), epoch)
    writer.add_scalar('Test_mAP', map, epoch)
    writer.add_scalar('Test_AP', ap, epoch)
    writer.add_scalar('Test_Recall', recall, epoch)
    if epoch % opt.log_step == 0:
        logging.info(
            'Test: Epoch: [{0}]\t'
            'Loss_avg {loss_avg:.4f} \t'
            'mAP {map:.4f}\t'
            'ap {ap: .4f}\t'
            'recall {recall:.4f}\t'
            'learning_rate {lr:.7f}'.format(
                epoch,
                loss_avg=running_loss / len(test_loader),
                map=map,
                ap=ap,
                recall=recall,
                lr=model.optimizer.state_dict()['param_groups'][0]['lr']))

    return map


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:  ## 保存最佳模型
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 10 epochs"""
    lr = opt.learning_rate * (1**(epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

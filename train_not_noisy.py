import torch.optim as optim
import torch
from utils.train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset import C3D_h5, PCN_pcd
from PCDDataset import PCDDataset


def train():
    logging.info(str(args))
    # metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse', "f1"]
    # metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse', "cd_hyp", "f1"]
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    # if args.dataset == 'pcn':
    #     dataset = PCN_pcd(args.pcnpath, prefix="train")
    #     dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    # elif args.dataset == 'c3d':
    #     dataset = C3D_h5(args.c3dpath, prefix="train")
    #     dataset_test = C3D_h5(args.c3dpath, prefix="val")
    # else:
    #     raise ValueError('dataset is not exist')
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                          shuffle=True, num_workers=int(args.workers))
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
    #                                               shuffle=False, num_workers=int(args.workers))
    # data_dir = './data'
    data_dir = './data_mix/data'
    
    dataset = PCDDataset(data_dir)
    dataloader= torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True
    )

    # data_dir = './data_test'
    data_dir = './data_mix/data_test'

    dataset_test = PCDDataset(data_dir)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=True
    )


    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))
    
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
    
    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]
    
    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)


    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
    
    for epoch in range(args.start_epoch, args.nepoch):
    
        train_loss_meter.reset()
        net.module.train()


        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
    
            inputs, gt = data["src_pcd"], data["model_pcd_transformed"]
            # mean_feature = None
    
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
    
            out2, loss2, net_loss = net(inputs, gt)
    
            train_loss_meter.update(net_loss.mean().item())
    
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
    
            optimizer.step()
    
            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss2.mean().item(), net_loss.mean().item(), lr))
    
        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")
    
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            inputs, gt = data["src_pcd"], data["model_pcd_transformed"]
            # mean_feature = None
    
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            # result_dict = net(inputs, gt, is_training=False, mean_feature=mean_feature)
            result_dict = net(inputs, gt, is_training=False)
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())
    
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)
    
        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)
    
        logging.info(curr_log)
        logging.info(best_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs',arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + args.dataset
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print('save_path:', args.work_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)],
                        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
                        force=True)
    train()




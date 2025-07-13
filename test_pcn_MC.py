import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.train_utils import *
from dataset import PCN_pcd
from PCDDataset import PCDDataset
import numpy as np
import torch
import torch.nn.functional as F


def save_obj(point, path):
    n = point.shape[0]
    with open(path, 'w') as f:
        for i in range(n):
            f.write("v {0} {1} {2}\n".format(point[i][0], point[i][1], point[i][2]))
    f.close()


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def test():
    data_dir = './data_test_unseen/'
    # data_dir = './data_sim/data_test/'
    dataset_test = PCDDataset(data_dir)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=True
    )
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()
    enable_dropout(net)  # Enable dropout for MC Dropout

    # metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse', "f1"]
    # metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse', "cd_hyp", "f1"]
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([8, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft']

    logging.info('Testing...')

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            inputs_cpu, gt = data['src_pcd'], data['model_pcd_transformed']
            gt = gt.float().cuda()
            inputs = inputs_cpu.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            # Standard forward pass
            result_dict = net(inputs, gt, is_training=False)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            # MC Dropout Passes with alignment
            T = 60
            all_out2 = []
            for _ in range(T):
                temp_result = net(inputs, gt, is_training=False)
                all_out2.append(temp_result['out2'].unsqueeze(0))
            all_out2 = torch.cat(all_out2, dim=0)  # (T, B, N, 3)

            # mean_out2 = all_out2.mean(dim=0)       # (B, N, 3)
            # std_out2 = all_out2.std(dim=0)         # (B, N, 3)

            # Nearest Neighbor Alignment
            ref_out2 = all_out2[0]  # Reference (B, N, 3)
            aligned_out2 = []
            for t in range(T):
                aligned_batch = []
                for b in range(all_out2.size(1)):
                    cur = all_out2[t, b]  # (N, 3)
                    ref = ref_out2[b]     # (N, 3)
                    dist = torch.cdist(ref.unsqueeze(0), cur.unsqueeze(0))  # (1, N, N)
                    idx = dist.argmin(dim=-1)  # (1, N)
                    aligned = cur[idx[0]]
                    aligned_batch.append(aligned)
                aligned_out2.append(torch.stack(aligned_batch))

            aligned_out2 = torch.stack(aligned_out2)  # (T, B, N, 3)
            mean_out2 = aligned_out2.mean(dim=0)      # (B, N, 3)
            std_out2 = aligned_out2.std(dim=0)        # (B, N, 3)

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                for j in range(args.batch_size):
                    out_dir = os.path.join(os.path.dirname(args.load_model), 'all')
                    os.makedirs(out_dir, exist_ok=True)

                    path = lambda name: os.path.join(out_dir, f'batch{i}_sample{j}_{name}.obj')
                    np_path = os.path.join(out_dir, f'batch{i}_sample{j}_data.npz')

                    # Optional .obj saving
                    # save_obj(result_dict['out2'][j], path('output'))
                    # save_obj(result_dict['out1'][j], path('output_inter'))
                    # save_obj(inputs.transpose(2, 1)[j], path('input'))
                    # save_obj(gt[j], path('gt'))

                    output_dict = {
                        'src_pcd': inputs.transpose(2, 1)[j].cpu().numpy(),
                        'src_pcd_inter': result_dict['out1'][j].cpu().numpy(),
                        'xyz': result_dict['out2'][j].cpu().numpy(),
                        'model_pcd_transformed': gt[j].cpu().numpy(),
                        'mean_out2': mean_out2[j].cpu().numpy(),
                        'std_out2': std_out2[j].cpu().numpy(),
                    }
                    np.savez(np_path, **output_dict)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs', arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, 'test.log')),
        logging.StreamHandler(sys.stdout)
    ])

    test()

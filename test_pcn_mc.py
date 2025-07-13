import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.train_utils import *
from dataset import PCN_pcd
import h5py
from PCDDataset import PCDDataset
import numpy as np

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def calculate_pose_errors(gt_poses, pred_poses):
    """
    Calculate rotation and translation errors between ground truth and predicted poses.
    
    Args:
        gt_poses: Ground truth poses [B, 4, 4] or dict with 'Rs_gt' and 'ts_gt'
        pred_poses: Predicted poses [B, 4, 4] or dict with 'Rs_pred' and 'ts_pred'
    
    Returns:
        dict: Contains rotation_error (degrees), translation_error (distance)
    """
    if isinstance(gt_poses, dict) and isinstance(pred_poses, dict):
        # Extract from dictionaries
        Rs_gt = gt_poses['Rs_gt']  # [B, 3, 3]
        ts_gt = gt_poses['ts_gt']  # [B, 3]
        Rs_pred = pred_poses['Rs_pred']  # [B, 3, 3]
        ts_pred = pred_poses['ts_pred']  # [B, 3]
    else:
        # Extract from 4x4 matrices
        Rs_gt = gt_poses[:, :3, :3]
        ts_gt = gt_poses[:, :3, 3]
        Rs_pred = pred_poses[:, :3, :3]
        ts_pred = pred_poses[:, :3, 3]
    
    batch_size = Rs_gt.shape[0]
    
    # Calculate rotation errors
    rotation_errors = []
    for i in range(batch_size):
        # Relative rotation: R_error = R_pred^T @ R_gt
        R_error = torch.matmul(Rs_pred[i].T, Rs_gt[i])
        
        # Convert to angle using trace formula: cos(theta) = (trace(R) - 1) / 2
        trace = torch.trace(R_error)
        cos_angle = (trace - 1) / 2
        
        # Clamp to avoid numerical issues
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        # Calculate angle in radians then convert to degrees
        angle_rad = torch.acos(cos_angle)
        angle_deg = angle_rad * 180.0 / torch.pi
        
        rotation_errors.append(angle_deg)
    
    # Calculate translation errors (Euclidean distance)
    translation_errors = torch.norm(ts_pred - ts_gt, dim=1)
    
    rotation_errors = torch.stack(rotation_errors)
    
    return rotation_errors, translation_errors

def save_h5(data, path):
    f = h5py.File(path, 'w')
    a = data.data.cpu().numpy()
    f.create_dataset('data', data=a)
    f.close()

def save_obj(point, path):
    n = point.shape[0]
    with open(path, 'w') as f:
        for i in range(n):
            f.write("v {0} {1} {2}\n".format(point[i][0],point[i][1],point[i][2]))
    f.close()

def test():
    # dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
    #                                               shuffle=False, num_workers=int(args.workers))
    data_dir = './data_files/data_fighter_test/'
    
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

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()
    enable_dropout(net)  # Enable dropout for MC Dropout

    metrics =['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse', "rotation", "translation", "cd_p_models", "cd_t_models"]
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([8, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft']

    logging.info('Testing...')

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            tgt_pose = data['pose']
            inputs_cpu, gt = data['src_pcd'], data['model_pcd_transformed']
            models = data['model_pcd']
            gt = gt.float().cuda()

            inputs = inputs_cpu.float().cuda()
            models.float().cuda()
            
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict = net(inputs, gt, models, is_training=False)
            for k, v in test_loss_meters.items():
                if k in result_dict:
                    v.update(result_dict[k].mean().item())

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


            
            if 'Rs_pred' in result_dict and 'ts_pred' in result_dict:
                # tgt_pose is [B, 4, 4] SE(3) matrices
                print(tgt_pose.shape)
                Rs_gt = tgt_pose[:, :3, :3].cuda()
                ts_gt = tgt_pose[:, :3, 3].cuda()
                Rs_pred = result_dict['Rs_pred']
                ts_pred = result_dict['ts_pred']
                
                rotation_error, translation_error = calculate_pose_errors(
                    {'Rs_gt': Rs_gt, 'ts_gt': ts_gt},
                    {'Rs_pred': Rs_pred, 'ts_pred': ts_pred}
                )
                
                test_loss_meters['rotation'].update(rotation_error.mean().item())
                test_loss_meters['translation'].update(translation_error.mean().item())
            # if i % args.step_interval_to_print == 0:
            #     logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            


            if args.save_vis:
                for j in range(args.batch_size):
                    if not os.path.isdir(os.path.join(os.path.dirname(args.load_model), 'all')):
                        os.makedirs(os.path.join(os.path.dirname(args.load_model), 'all'))
                        
                    path = os.path.join(os.path.dirname(args.load_model), 'all', f'batch{i}_sample{j}_output.obj')
                    path_t = os.path.join(os.path.dirname(args.load_model), 'all',f'batch{i}_sample{j}_output_inter.obj')
                    path_input = os.path.join(os.path.dirname(args.load_model), 'all',f'batch{i}_sample{j}_input.obj')
                    path_gt = os.path.join(os.path.dirname(args.load_model), 'all',f'batch{i}_sample{j}_gt.obj')

                    # save_obj(result_dict['out2'][j], path)
                    # save_obj(result_dict['out1'][j], path_t)
                    # # Save input pointcloud (need to transpose back to original format)
                    # save_obj(inputs.transpose(2, 1)[j], path_input)

                    # save_obj(gt[j], path_gt)
                    # path_src_inter = os.path.join(os.path.dirname(args.load_model), 'all', f'batch{i}_sample{j}_src_inter.obj')
                    # save_obj(inputs.transpose(2, 1)[j], path_src_inter)
                    # Save data in numpy format
                    pose = np.eye(4)
                    if 'Rs_pred' in result_dict and 'ts_pred' in result_dict:
                        pose[:3, :3] = result_dict['Rs_pred'][j].cpu().numpy()
                        pose[:3, 3] = result_dict['ts_pred'][j].cpu().numpy()
                    output_dict = {
                        'src_pcd': inputs.transpose(2, 1)[j].cpu().numpy(),
                        'tgt_pcd': gt[j].cpu().numpy(),
                        'tgt_pose': tgt_pose[j].cpu().numpy(),

                        'pred_pcd_coarse': result_dict['out1'][j].cpu().numpy(),
                        'pred_pcd': mean_out2[j].cpu().numpy(),                        
                        'pred_models': result_dict['models_transformed'][j].cpu().numpy() if 'models_transformed' in result_dict else None,
                        'pred_pose': pose if 'Rs_pred' in result_dict and 'ts_pred' in result_dict else None,
                        

                        
                        
                        
                    }
                    np_path = os.path.join(os.path.dirname(args.load_model), 'all', f'batch{i}_sample{j}_data.npz')
                    np.savez(np_path, **output_dict)
        # logging.info('Loss per category:')
        # category_log = ''
        # for i in range(8):
        #     category_log += '\ncategory name: %s' % (cat_name[i])
        #     for ind, m in enumerate(metrics):
        #         scale_factor = 1 if m == 'f1' else 10000
        #         category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i] * scale_factor)
        # logging.info(category_log)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            if metric in result_dict:
                overview_log += '%s: %f ' % (metric, meter.avg)
        
        overview_log += '\n'
        if 'Rs_pred' in result_dict and 'ts_pred' in result_dict:
            overview_log += 'rotation: %f translation: %f ' % (test_loss_meters['rotation'].avg, test_loss_meters['translation'].avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs',arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()

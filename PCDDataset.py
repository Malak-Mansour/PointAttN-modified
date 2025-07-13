import torch, os, numpy as np

class PCDDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, augment=False, aug_params=None):
        self.data_files = [f for f in os.listdir(data_dir) if f.startswith('processed_dataset_')]
        self.data_dir = data_dir
        self.augment = augment
        
        # Default augmentation parameters
        self.aug_params = {
            'noise_points': 8,
            'noise_range': (-0.2, 0.2),
            'random_rotate': True,
            'random_jitter': True,
            'jitter_sigma': 0.01,
            'random_scale': (0.8, 1.2)
        }
        
        # Update with provided parameters
        if aug_params:
            self.aug_params.update(aug_params)
    
    def __len__(self):
        return len(self.data_files)
    
    def augment_pointcloud(self, src_pcd, model_pcd_transformed):
        """Apply augmentation to point clouds"""
        src_pcd = src_pcd.copy()
        model_pcd_transformed = model_pcd_transformed.copy()
        
        # Add noise to random points
        if self.aug_params['noise_points'] > 0:
            num_points = len(src_pcd)
            if num_points >= self.aug_params['noise_points']:
                random_indices = np.random.choice(num_points, self.aug_params['noise_points'], replace=False)
                noise_range = self.aug_params['noise_range']
                noise = np.random.uniform(noise_range[0], noise_range[1], 
                                         (self.aug_params['noise_points'], 3))
                src_pcd[random_indices] += noise
        
        # Random rotation around y-axis
        if self.aug_params['random_rotate']:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_theta, sin_theta = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])
            src_pcd = np.dot(src_pcd, rotation_matrix)
            model_pcd_transformed = np.dot(model_pcd_transformed, rotation_matrix)
        
        # Add small jitter to all points
        if self.aug_params['random_jitter']:
            src_pcd += np.random.normal(0, self.aug_params['jitter_sigma'], size=src_pcd.shape)
        
        # Random scaling
        if self.aug_params['random_scale']:
            scale = np.random.uniform(self.aug_params['random_scale'][0], 
                                     self.aug_params['random_scale'][1])
            src_pcd *= scale
            model_pcd_transformed *= scale
        
        return src_pcd, model_pcd_transformed
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)
        
        src_pcd = data['src_pcd']
        model_pcd_transformed = data['model_pcd_transformed']
        
        # Apply augmentation if enabled
        if self.augment:
            src_pcd, model_pcd_transformed = self.augment_pointcloud(src_pcd, model_pcd_transformed)
          
        
        return {
            'src_pcd': torch.from_numpy(src_pcd).float(),
            'model_pcd_transformed': torch.from_numpy(model_pcd_transformed).float(),
        }
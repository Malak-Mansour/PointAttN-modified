import torch, os, numpy as np

# Create train loader from processed dataset files
class PCDDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, augment=False, mask_ratio=0.1, max_rotation=15, max_translation=5, num_points=512):
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        if len(self.data_files) == 0:   
            self.data_files = [f for f in os.listdir(data_dir)]
            print(f"Loaded other data files length: {len(self.data_files)}")
        self.data_dir = data_dir
        self.augment = augment
        self.mask_ratio = mask_ratio  # Ratio of points to mask
        self.max_rotation = max_rotation  # Max rotation in degrees
        self.max_translation = max_translation  # Max translation as fraction of point cloud size
        self.num_points = num_points  # Fixed number of points to sample
        
        self.first_time = True
        self.model = None  # Class variable to store the first model point cloud
            
    def __len__(self):
        return len(self.data_files)
    
    def random_rotation_matrix(self):
        """Generate a random 3D rotation matrix"""
        angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=3) * np.pi / 180
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        return np.dot(Rz, np.dot(Ry, Rx))
    
    def random_mask(self, points):
        """Randomly remove a percentage of points"""
        num_points = points.shape[0]
        keep_mask = np.random.choice([True, False], size=num_points, p=[1-self.mask_ratio, self.mask_ratio])
        return points[keep_mask]
    
    def random_slice(self, points):
        """Randomly slice the point cloud along one axis"""
        axis = np.random.randint(0, 3)  # Choose x, y, or z axis
        mid = (points[:, axis].max() + points[:, axis].min()) / 2
        direction = np.random.choice([-1, 1])
        if direction > 0:
            mask = points[:, axis] > mid
        else:
            mask = points[:, axis] < mid
        return points[mask]
    
    def farthest_point_sample_numpy(self, points, npoint):
        """
        Pure numpy implementation of farthest point sampling
        Input:
            points: pointcloud data, [N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud indices, [npoint]
        """
        N, D = points.shape
        centroids = np.zeros(npoint, dtype=np.int64)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        
        for i in range(npoint):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
            
        return centroids
    
    def sample_points_fps(self, points, n):
        """Sample exactly n points using numpy FPS implementation"""
        if len(points) == 0:
            # Handle empty point cloud by returning zeros
            return np.zeros((n, 3))
            
        # Convert tensor to numpy if needed
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = points
            
        # If we have fewer than n points, duplicate points before FPS
        if points_np.shape[0] < n:
            # Calculate how many times to repeat
            repeat_factor = (n // points_np.shape[0]) + 1
            points_np = np.tile(points_np, (repeat_factor, 1))
            points_np = points_np[:n, :]
        
        # Apply farthest point sampling
        fps_indices = self.farthest_point_sample_numpy(points_np, n)
        sampled_points = points_np[fps_indices]
        
        # Return in the same format as input
        if isinstance(points, torch.Tensor):
            return torch.from_numpy(sampled_points).float().to(points.device)
        else:
            return sampled_points
    
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)
        src_pcd = data['src_pcd']
        model_pcd_transformed = data['model_pcd_transformed']
        pose = data["pose"]
        
        if self.model is None:
            pose_inv = np.linalg.inv(pose)
            R = pose_inv[:3, :3]
            t = pose_inv[:3, 3]            
            self.model = np.dot(model_pcd_transformed, R.T) + t
        
        # R = pose[:3, :3]
        # t = pose[:3, 3]            
        # x = np.dot(self.model, R.T) + t
        # print(np.linalg.norm(x - model_pcd_transformed).round(3))
        
        aug_applied = False
        if self.augment:
            # Apply same geometric transformations to both point clouds
            if np.random.random() < 0.5:
                R = self.random_rotation_matrix()
                t = np.random.uniform(-self.max_translation, self.max_translation, size=3)
                t *= np.max(np.abs(src_pcd.max(axis=0) - src_pcd.min(axis=0)))
                
                # Apply same transformation to both point clouds
                src_pcd = np.dot(src_pcd, R.T) + t
                model_pcd_transformed = np.dot(model_pcd_transformed, R.T) + t
                # Compose the SE(3) transformation
                T_aug = np.eye(4)
                T_aug[:3, :3] = R
                T_aug[:3, 3] = t
                pose = T_aug @ pose  # or np.dot(T_aug, pose)
                
                # # Optionally apply structural modifications only to source
                # if np.random.random() > 0.5:
                #     src_pcd = self.random_mask(src_pcd)
                # else:
                #     src_pcd = self.random_slice(src_pcd)
                    
                # # Check if we have enough points
                # min_points = 512
                # if src_pcd.shape[0] < min_points:
                #     # Reapply just the geometric transformation
                #     src_pcd = np.dot(data['src_pcd'], R.T) + t
                aug_applied = True
        
        # Use FPS to sample exactly num_points points from source
        # src_pcd = self.sample_points_fps(src_pcd, self.num_points)
                # Save point clouds to debug directory
        
        # src_pcd *= 1000
        # model_pcd_transformed *= 1000
        # if aug_applied:
        #     print(f"Augmentation applied to {self.data_files[idx]}")
        #     debug_dir = os.path.join(os.path.dirname(self.data_dir), 'debug_pcds')
        #     os.makedirs(debug_dir, exist_ok=True)
        #     np.savez_compressed(os.path.join(debug_dir, f'debug_{os.path.basename(self.data_files[idx])}'), 
        #                     src_pcd=src_pcd.cpu().numpy() if isinstance(src_pcd, torch.Tensor) else src_pcd,
        #                     model_pcd_transformed=model_pcd_transformed)        
        return {
            'src_pcd': torch.from_numpy(src_pcd).float() if isinstance(src_pcd, np.ndarray) else src_pcd,
            'model_pcd_transformed': torch.from_numpy(model_pcd_transformed).float(),
            'pose': torch.from_numpy(pose).float(),
            'model_pcd': torch.from_numpy(self.model).float() if self.model is not None else None
        }
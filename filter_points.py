import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
import os
import glob
import open3d as o3d
model_path = 'yolo_model.pt'
yolo_model = YOLO(model_path)
# sam2 setup        
checkpoint =  "sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Create a pipeline for bag playback
import pyrealsense2 as rs

import torch
import threading
from sensor_msgs.msg import PointCloud2, PointField
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import numpy as np






def segment_pointcloud(frame, points):
    # YOLO detection and SAM2 segmentation (using existing code)
    results = yolo_model.predict(source=frame, conf=0.5, verbose=True)
    if len(results[0].boxes) == 0:
        print("No object detected")
        return None
    else:
        detection_results = {}
        masks_list = []
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy().astype(float)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(frame)
            for i, box in enumerate(bboxes):
                input_box = np.array(box).reshape(1, 4)
                masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
                mask = (masks > 0).astype(np.uint8) * 255
                masks_list.append(mask[0])

        # Create a composite visualization (optional; not used in output)
        visualization_frame = frame.copy()
        for i, mask in enumerate(masks_list):
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = [0, 255, 0]  # Green overlay
            visualization_frame = cv2.addWeighted(visualization_frame, 1, colored_mask, 0.5, 0)
            # Annotate with index (optional)
            coords = np.column_stack(np.where(mask > 0))
            if coords.size > 0:
                y, x = coords.mean(axis=0).astype(int)
                cv2.putText(visualization_frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite("segmented_visualization.png", visualization_frame)
        print("Segmented image saved as segmented_visualization.png")
        print(f"Number of detections: {len(masks_list)}")

        # Prepare output: each detection has filtered points and its own segmented image
        detection_results = {}
        # Reshape points to align with the image dimensions (assumes image shape is 480 x 848)
        points_reshaped = points.reshape(480, 848, 3)
        for idx, mask in enumerate(masks_list):
            # Extract point cloud for detection: only points where the mask is active
            detection_points = points_reshaped[mask > 0]
            # Create a segmented image: zero out areas not in the mask
            detection_image = np.zeros_like(frame)
            detection_image[mask > 0] = frame[mask > 0]
            detection_results[f"detection_{idx}"] = {"points": detection_points, "image": detection_image}

        return detection_results

def extra_filter(points):
    if points.shape[0] > 0:
        # Compute centroid and distances to it
        
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        # Use median absolute deviation to set a robust threshold
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
            # Set threshold as median + 3 * MAD (avoid zero MAD)
        threshold = median_dist + 3 * (mad if mad > 0 else 1e-6)
            # Keep only inliers that are not too far from the centroid
        inliers = points[distances <= threshold]
        print(f"number of inliers {len(inliers)}")
        if len(inliers) > 0:
            filtered_points = inliers.copy()
            return filtered_points
    return None


bag_file = 'bags/20250218_201242.bag'

# Initialize the pipeline and configure it to read from a .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)

pipeline.start(config)

# Get the depth sensor's depth scale
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Frame counter for saving images
# Instead, create and visualize a point cloud every 10 frames using the RealSense point cloud object.
pc = rs.pointcloud()  # RealSense point cloud object

frame_counter = 0
while True:
    try:
        # Wait for frames (timeout to catch end of bag file)
        frames = pipeline.wait_for_frames(timeout_ms=5000)
    except RuntimeError:
        print("End of bag file reached.")
        break

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("No frames found, finishing frame extraction.")
        break

    frame_counter += 1

    if frame_counter % 2== 0:
        # ----- Depth Image Filtering (Median Blur) -----
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_filtered = cv2.medianBlur(depth_image, 5)

        # ----- Point Cloud Generation -----
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        pts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # Instead of filtering out points, set points outside threshold to [0,0,0]
        valid_mask = pts[:, 2] < 2.0
        pts = np.where(valid_mask[:, None], pts, np.zeros_like(pts))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # ----- Voxel Downsampling with Replacement -----
        voxel_size = 0.05
        pts_array = np.asarray(pcd.points)
        voxel_indices = np.floor(pts_array / voxel_size).astype(np.int32)
        _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        min_points_per_voxel = 30
        voxel_mask = counts[inverse] >= min_points_per_voxel
        pts_array = np.where(voxel_mask[:, None], pts_array, np.zeros_like(pts_array))
        pcd.points = o3d.utility.Vector3dVector(pts_array)

        # ----- Bounding Box Filter with Replacement -----
        pts_np = np.asarray(pcd.points)
        bbox_mask = np.all((pts_np >= -1.0) & (pts_np <= 1.0), axis=1)
        pts_np = np.where(bbox_mask[:, None], pts_np, np.zeros_like(pts_np))
        
       





        pcd.points = o3d.utility.Vector3dVector(pts_np)
       
        
        # ----- Visualize and Save the Depth Image with Improved Color Mapping -----
        # Normalize depth image to 8-bit for color mapping
        depth_norm = cv2.normalize(depth_image_filtered, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        colored_depth_image = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite("colored_depth_image.png", colored_depth_image)

        # ----- Visualize the RGB Image with Correct Color Channels -----
        rgb_image = np.asanyarray(color_frame.get_data())
        # Convert from RGB to BGR for proper display with OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("corrected_rgb_image.png", bgr_image)

        segmented_points = segment_pointcloud(np.array(bgr_image), pts_np)  # should have the same shape as original pcd
        if segmented_points is None:
            print("No segmented points found.")
            continue

        # Combine all segmented points from each detection into one point cloud
        combined_points_list = []
        for detection, data in segmented_points.items():
            pts_detection = data["points"]
            if pts_detection.size > 0:
                combined_points_list.append(pts_detection)
        if combined_points_list:
            combined_points = np.concatenate(combined_points_list, axis=0)
        else:
            print("No valid points in segmentation.")
            combined_points = np.empty((0, 3), dtype=pts_np.dtype)

        # Remove the segmented (combined) points from the original pts_np.
        # We use a structured view to perform set difference.
        def to_structured(arr):
            return np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))

        structured_pts_np = to_structured(pts_np)
        structured_combined = to_structured(combined_points)
        remaining_mask = ~np.in1d(structured_pts_np, structured_combined)
        remaining_points = pts_np[remaining_mask]

        # Create two point clouds for visualization.
        pcd_remaining = o3d.geometry.PointCloud()
        pcd_remaining.points = o3d.utility.Vector3dVector(remaining_points)
        pcd_remaining.paint_uniform_color([1, 1, 0])  # Yellow for non-segmented points

        pcd_combined = o3d.geometry.PointCloud()
        pcd_combined.points = o3d.utility.Vector3dVector(combined_points)
        pcd_combined.paint_uniform_color([1, 0, 0])  # Red for combined segmented points

        visu = o3d.visualization.Visualizer()
        visu.create_window(
            window_name="Segmented (Red) and Remaining (Yellow) Points", width=800, height=600
        )
        visu.add_geometry(pcd_remaining)
        visu.add_geometry(pcd_combined)
        opt = visu.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray similar to rviz dark
        visu.run()
        visu.destroy_window()

        def show_pointcloud(points, window_title):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_title)
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()

        print(f"segmented_points keys {segmented_points.keys()}")
        output_folder_PCD = "data/segmented"
        output_folder_img = "data/images"
        for detection, data in segmented_points.items():
            filtered_points = extra_filter(data["points"])
            image = data["image"]
            # Save the segmented image using the frame and detection index format _0000_2
            image_filename = f"segmented_image_{frame_counter:04d}_{detection}.png"
            image_path = os.path.join(output_folder_img, image_filename)
            cv2.imwrite(image_path, image)
            if filtered_points is not None:
                #threading.Thread(target=show_pointcloud, args=(filtered_points, detection), daemon=True).start()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(filtered_points)
                points_filename = f"filtered_points_{frame_counter:04d}_{detection}.ply"
                points_path = os.path.join(output_folder_PCD, points_filename)
                o3d.io.write_point_cloud(points_path, pcd)
                # cv2.imshow(detection, data["image"])
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break
       
       
        # cv2.imshow("RGB Image", bgr_image)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        

print("Finished processing bag file.")
pipeline.stop()
cv2.destroyAllWindows()

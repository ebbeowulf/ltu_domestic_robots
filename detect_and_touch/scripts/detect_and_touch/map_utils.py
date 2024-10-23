from sklearn.cluster import DBSCAN
from farthest_point_sampling.fps import farthest_point_sampling
import numpy as np
import open3d as o3d
import pickle
from camera_params import camera_params
import torch
import os

DEVICE = torch.device("cpu")

DBSCAN_MIN_SAMPLES=20 
DBSCAN_GRIDCELL_SIZE=0.01
DBSCAN_EPS=0.018 # allows for connections in cells full range of surrounding cube DBSCAN_GRIDCELL_SIZE*2.5
CLUSTER_MIN_COUNT=10000
CLUSTER_PROXIMITY_THRESH=0.3
CLUSTER_TOUCHING_THRESH=0.05

## Calculate the iou between two bounding boxes
def calculate_iou(minA,maxA,minB,maxB):
    deltaA=maxA-minA
    areaA=np.prod(deltaA)
    deltaB=maxB-minB
    areaB=np.prod(deltaB)
    isct_min=np.vstack((minA,minB)).max(0)
    isct_max=np.vstack((maxA,maxB)).min(0)
    isct_delta=isct_max-isct_min
    if (isct_delta<=0).sum()>0:
        return 0.0
    areaI=np.prod(isct_delta)
    return areaI / (areaA + areaB - areaI)

## Extract distinct clusters from a point cloud using DBSCAN
def get_distinct_clusters(pcloud, gridcell_size=DBSCAN_GRIDCELL_SIZE, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, cluster_min_count=CLUSTER_MIN_COUNT, floor_threshold=0.1):
    clouds=[]
    if pcloud is None or len(pcloud.points)<cluster_min_count:
        return clouds
    if gridcell_size>0:
        pcd_small=pcloud.voxel_down_sample(gridcell_size)
        pts=np.array(pcd_small.points)
        p2=DBSCAN(eps=eps, min_samples=min_samples,n_jobs=10).fit(pts)
    else:
        pts=np.array(pcloud.points)
        p2=DBSCAN(eps=eps, min_samples=min_samples,n_jobs=10).fit(pts)

    # Need to get the cluster sizes... so we can focus on the largest clusters only
    cl_cnt=np.array([ (p2.labels_==cnt).sum() for cnt in range(p2.labels_.max() + 1) ])
    validID=np.where(cl_cnt>cluster_min_count)[0]
    if validID.shape[0]>0:
        sortedI=np.argsort(-cl_cnt[validID])

        for id in validID[sortedI][:10]:
            whichP=(p2.labels_==id)
            pts2=pts[whichP]
            whichP2=(pts2[:,2]>floor_threshold)
            if whichP2.sum()>cluster_min_count:
                clouds.append(object_pcloud(pts2[whichP2]))

    return clouds

# A wrapper function for get_distinct_clusters - builds the pcloud, then the clusters, and then compresses the results for transmission
def create_object_clusters(pts_xyz, pts_prob, floor_threshold=-0.1, detection_threshold=0.5, min_cluster_points=100, compress_clusters=True):
    if pts_xyz.shape[0]==0 or pts_xyz.shape[0]!=pts_prob.shape[0]:
        return []
    whichP=(pts_prob>=detection_threshold)
    pcd=o3d.geometry.PointCloud()
    xyzF=pts_xyz[whichP]
    F2=np.where(np.isnan(xyzF).sum(1)==0)
    xyzF2=xyzF[F2]        
    pcd.points=o3d.utility.Vector3dVector(xyzF2)
    object_clusters=get_distinct_clusters(pcd, floor_threshold=floor_threshold, cluster_min_count=min_cluster_points)

    probF=pts_prob[whichP]
    probF2=probF[F2]
    for idx in range(len(object_clusters)):
        object_clusters[idx].estimate_probability(xyzF2,probF2)
        if compress_clusters:
            object_clusters[idx].compress_object()

    return object_clusters

class object_pcloud():
    def __init__(self, pts, label:str=None, num_samples=1000):
        self.box=np.vstack((pts.min(0),pts.max(0)))
        self.pts=pts
        self.pts_shape=self.pts.shape
        self.label=label
        self.farthestP=farthest_point_sampling(self.pts, num_samples)
        self.prob_stats=None
        self.centroid=self.pts.mean(0)

    def set_label(self, label):
        self.label=label
    
    def is_box_overlap(self, input_cloud, dimensions=[0,1,2], threshold=0.3):
        for dim in dimensions:
            if self.box[1,dim]<(input_cloud.box[0,dim]-threshold) or self.box[0,dim]>=(input_cloud.box[1,dim]+threshold):
                return False
        return True

    def compute_cloud_distance(self, input_cloud):
        input_pt_matrix=input_cloud.pts[input_cloud.farthestP]
        min_sq_dist=1e10
        for pid in self.farthestP:
            min_sq_dist=min(min_sq_dist, ((input_pt_matrix-self.pts[pid])**2).sum(1).min())
        return np.sqrt(min_sq_dist)
    
    def is_above(self, input_cloud):
        # Should be overlapped in x + y directions
        if self.centroid[0]>input_cloud.box[0,0] and self.centroid[0]<input_cloud.box[1,0] and self.centroid[1]>input_cloud.box[0,1] and self.centroid[1]<input_cloud.box[1,1]:
            # Should also be "above" the other centroid
            return self.centroid[2]>input_cloud.centroid[2]
        return False
    
    def estimate_probability(self, original_xyz, original_prob):
        filt=(original_xyz[:,0]>=self.box[0][0])*(original_xyz[:,0]<=self.box[1][0])*(original_xyz[:,1]>=self.box[0][1])*(original_xyz[:,1]<=self.box[1][1])*(original_xyz[:,2]>=self.box[0][2])*(original_xyz[:,2]<=self.box[1][2])
        self.prob_stats=dict()
        self.prob_stats['max']=original_prob[filt].max()
        self.prob_stats['mean']=original_prob[filt].mean()
        self.prob_stats['stdev']=original_prob[filt].std()
        self.prob_stats['pcount']=filt.shape[0]
    
    def size(self):
        return self.pts_shape[0]
    
    def compress_object(self):
        self.pts=None
        self.farthestP=None

class pcloud_from_images():
    def __init__(self, params:camera_params):
        self.params=params
        self.YS=None
        self.rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
        self.cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)
        self.rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)        
        self.loaded_image=None

    # # Image loading to allow us to process more than one class in rapid succession
    # def load_image_from_file(self, fList:rgbd_file_list, image_key, max_distance=10.0):
    #     colorI=cv2.imread(fList.get_color_fileName(image_key), -1)
    #     depthI=cv2.imread(fList.get_depth_fileName(image_key), -1)
    #     poseM=fList.get_pose(image_key)
    #     self.load_image(colorI, depthI, poseM, image_key, max_distance=max_distance)

    def load_image(self, colorI:np.ndarray, depthI:np.ndarray, poseM:np.ndarray, uid_key:str, max_distance=10.0):
        if self.loaded_image is None or self.loaded_image['key']!=uid_key:
            try:
                if self.loaded_image is None:
                    self.loaded_image=dict()
                self.loaded_image['depthT']=torch.tensor(depthI.astype('float')/1000.0,device=DEVICE)
                self.loaded_image['colorT']=torch.tensor(colorI,device=DEVICE)
                self.loaded_image['x'] = self.cols*self.loaded_image['depthT']/self.params.fx
                self.loaded_image['y'] = self.rows*self.loaded_image['depthT']/self.params.fy
                self.loaded_image['depth_mask']=(self.loaded_image['depthT']>1e-4)*(self.loaded_image['depthT']<max_distance)

                # Build the rotation matrix
                self.loaded_image['M']=torch.matmul(self.rot_matrixT,torch.tensor(poseM,device=DEVICE))

                # Save the key last so we can skip if called again
                self.loaded_image['key']=uid_key

                print(f"Image loaded: {uid_key}")
                return True
            except Exception as e:
                print(f"Failed to load image materials for {uid_key}")
                self.loaded_image=None
            return False
        return True

    def get_rotated_points(self, filtered_maskT):
        pts=torch.stack([self.loaded_image['x'][filtered_maskT],
                            self.loaded_image['y'][filtered_maskT],
                            self.loaded_image['depthT'][filtered_maskT],
                            torch.ones(((filtered_maskT>0).sum()),
                            device=DEVICE)],dim=1)
        pts_rot=torch.matmul(self.loaded_image['M'],pts.transpose(0,1))
        return pts_rot[:3,:].transpose(0,1)


    def get_pts_per_class(self, tgt_class, use_connected_components=False, rotate90=False):
        # Build the class associated mask for this image
        cls_mask=self.YS.get_mask(tgt_class)
        if rotate90:
            cls_maskT=torch.tensor(np.rot90(cls_mask,axes=(0,1)).copy(),device=DEVICE)
        else:
            cls_maskT=torch.tensor(cls_mask,device=DEVICE)

        # Apply connected components if requested       
        if use_connected_components:
            filtered_maskT=self.cluster_pcloud()
        else:
            filtered_maskT=cls_maskT*self.loaded_image['depth_mask']

        # Return all points associated with the target class
        pts_rot=self.get_rotated_points(filtered_maskT) 
        if rotate90:
            probs=np.rot90(self.YS.get_prob_array(tgt_class),axes=(0,1))
            return {'xyz': pts_rot.cpu().numpy(), 
                    'rgb': self.loaded_image['colorT'][filtered_maskT].cpu().numpy(), 
                    'probs': probs[filtered_maskT]}
        else:
            return {'xyz': pts_rot.cpu().numpy(), 
                    'rgb': self.loaded_image['colorT'][filtered_maskT].cpu().numpy(), 
                    'probs': self.YS.get_prob_array(tgt_class)[filtered_maskT]}

    # def process_image(self, image_key, tgt_class, conf_threshold):
    def process_image(self, tgt_class, detection_threshold, segmentation_save_file=None):
        # Create the image segmentation file
        if self.YS is None or tgt_class not in self.YS.get_all_classes():
            from clip_segmentation import clip_seg
            self.YS=clip_seg([tgt_class])

        # Recover the segmentation file
        if segmentation_save_file is not None and os.path.exists(segmentation_save_file):
            if not self.YS.load_file(segmentation_save_file,threshold=detection_threshold):
                return None
        else:
            self.YS.process_image_numpy(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)    
            # self.YS.set_data(outputs,self.loaded_image['colorT'].size(),threshold=detection_threshold)

        return self.get_pts_per_class(tgt_class)

    def multi_prompt_process(self, prompts:list, detection_threshold, rotate90:False):
        if self.YS is None or prompts[0] not in self.YS.get_all_classes():
            from clip_segmentation import clip_seg
            self.YS=clip_seg(prompts)

        if rotate90:
            rot_color=np.rot90(self.loaded_image['colorT'].cpu().numpy(), k=1, axes=(1,0))
            self.YS.process_image_numpy(rot_color, detection_threshold)    
        else:
            self.YS.process_image_numpy(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)    

        all_pts=dict()
        # Build the class associated mask for this image
        for tgt_class in prompts:
            all_pts[tgt_class]=self.get_pts_per_class(tgt_class, rotate90=rotate90)

        return all_pts
    
    #Apply clustering - slow... probably in need of repair
    # def cluster_pclouds(self, image_key, tgt_class, cls_mask, threshold):
    #     save_fName=self.fList.get_class_pcloud_fileName(image_key,tgt_class)
    #     if os.path.exists(save_fName):
    #         with open(save_fName, 'rb') as handle:
    #             filtered_maskT=pickle.load(handle)
    #     else:
    #         # We need to build the boxes around clusters with clip-based segmentation
    #         #   YOLO should already have the boxes in place
    #         if self.YS.get_boxes(tgt_class) is None or len(self.YS.get_boxes(tgt_class))==0:
    #             self.YS.build_dbscan_boxes(tgt_class,threshold=threshold)
    #         # If this is still zero ...
    #         if len(self.YS.get_boxes(tgt_class))<1:
    #             return None
    #         combo_mask=(torch.tensor(cls_mask,device=DEVICE)>threshold)*self.loaded_image['depth_mask']
    #         # Find the list of boxes associated with this object
    #         boxes=self.YS.get_boxes(tgt_class)
    #         filtered_maskT=None
    #         for box in boxes:
    #             # Pick a point from the center of the mask to use as a centroid...
    #             ctrRC=get_center_point(self.loaded_image['depthT'], combo_mask, box[1])
    #             if ctrRC is None:
    #                 continue

    #             maskT=connected_components_filter(ctrRC,self.loaded_image['depthT'], combo_mask, neighborhood=10)
    #             # Combine masks from multiple objects
    #             if filtered_maskT is None:
    #                 filtered_maskT=maskT
    #             else:
    #                 filtered_maskT=filtered_maskT*maskT
    #         with open(save_fName,'wb') as handle:
    #             pickle.dump(filtered_maskT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     return filtered_maskT

    # Process a sequence of images, combining the resulting point clouds together
    #   The combined result will be saved to disk for faster retrieval
    # def process_fList(self, fList:rgbd_file_list, tgt_class, conf_threshold, use_connected_components=False):
    #     save_fName=fList.get_combined_raw_fileName(tgt_class)
    #     pcloud=None
    #     if os.path.exists(save_fName):
    #         try:
    #             with open(save_fName, 'rb') as handle:
    #                 pcloud=pickle.load(handle)
    #         except Exception as e:
    #             pcloud=None
    #             print("Failed to load save file - rebuilding... " + save_fName)
        
    #     if pcloud is None:
    #         # Build the pcloud from individual images
    #         pcloud={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8),'probs': []}
            
    #         image_key_list=clip_threshold_evaluation(fList, [tgt_class], conf_threshold)
    #         for key in image_key_list:
    #             self.load_image_from_file(fList, key)
    #             icloud=self.process_image(tgt_class, conf_threshold, use_connected_components=use_connected_components, segmentation_save_file=fList.get_segmentation_fileName(key, False, tgt_class))
    #             if icloud is not None and icloud['xyz'].shape[0]>100:
    #                 pcloud['xyz']=np.vstack((pcloud['xyz'],icloud['xyz']))
    #                 pcloud['rgb']=np.vstack((pcloud['rgb'],icloud['rgb']))
    #                 pcloud['probs']=np.hstack((pcloud['probs'],icloud['probs']))
            
    #         # Now save the result so we don't have to keep processing this same cloud
    #         with open(save_fName,'wb') as handle:
    #             pickle.dump(pcloud, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #     # Finally - filter the cloud with the requested confidence threshold
    #     whichP=(pcloud['probs']>conf_threshold)
    #     return {'xyz':pcloud['xyz'][whichP],'rgb':pcloud['rgb'][whichP],'probs':pcloud['probs'][whichP]}

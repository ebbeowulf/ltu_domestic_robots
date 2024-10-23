from sklearn.cluster import DBSCAN
from farthest_point_sampling.fps import farthest_point_sampling
import numpy as np
import open3d as o3d

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
def create_object_clusters(pts_xyz, pts_prob, floor_threshold=-0.1, detection_threshold=0.5, min_cluster_points=ABSOLUTE_MIN_CLUSTER_SIZE, compress_clusters=True):
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
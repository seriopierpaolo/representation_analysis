import torch
import numpy as np
from torch import nn



def filter_pointcloud(pointcloud, point_cloud_range):
    
    x, y, z, i = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], pointcloud[:, 3]

    # Filter points within the specified ranges
    mask = (x > point_cloud_range[0]) & (x < point_cloud_range[3]) & (y > point_cloud_range[1]) & (y < point_cloud_range[4]) \
        & (z > point_cloud_range[2]) & (z < point_cloud_range[5])
    x_filtered, y_filtered, z_filtered, i_filtered = x[mask], y[mask], z[mask], i[mask]

    points = np.stack([x_filtered, y_filtered, z_filtered, i_filtered], axis=1)

    return points


import torch
from torch import nn

def voxelize_batch(batch_points, point_cloud_range, voxel_size, max_points_per_pillar, max_pillars):
    B = len(batch_points)
    grid_x = int((point_cloud_range[3]-point_cloud_range[0]) / voxel_size[0])
    grid_y = int((point_cloud_range[4]-point_cloud_range[1]) / voxel_size[1])
    pillars, coords, batch_idx = [], [], []

    for b in range(B):
        pts = batch_points[b]
        mask = (
            (pts[:,0]>=point_cloud_range[0]) & (pts[:,0]<point_cloud_range[3]) &
            (pts[:,1]>=point_cloud_range[1]) & (pts[:,1]<point_cloud_range[4])
        )
        pts = pts[mask]
        x = ((pts[:,0]-point_cloud_range[0]) / voxel_size[0]).floor().long()
        y = ((pts[:,1]-point_cloud_range[1]) / voxel_size[1]).floor().long()
        idx = torch.stack([x,y],1)
        uniq, inv = torch.unique(idx, dim=0, return_inverse=True)
        for i,u in enumerate(uniq[:max_pillars]):
            sel = pts[inv==i][:max_points_per_pillar]
            mean = sel[:,:3].mean(0)
            cx, cy = (u[0]+0.5)*voxel_size[0]+point_cloud_range[0], (u[1]+0.5)*voxel_size[1]+point_cloud_range[1]
            f = torch.cat([sel,
                           sel[:,:3]-mean,
                           torch.stack([sel[:,0]-cx, sel[:,1]-cy],1)],1)
            pad = torch.zeros(max_points_per_pillar-sel.shape[0], f.shape[1], device=f.device)
            pillars.append(torch.cat([f,pad]))
            coords.append(u)
            batch_idx.append(b)
    return torch.stack(pillars), torch.stack(coords), torch.tensor(batch_idx)

class PillarFeatureNet(nn.Module):
    def __init__(self, in_ch=9, hidden=[64,128,256]):
        super().__init__()
        layers = []
        c = in_ch
        for h in hidden:
            layers += [nn.Linear(c,h), nn.BatchNorm1d(h), nn.ReLU()]
            c = h
        self.mlp = nn.Sequential(*layers)
        self.out_ch = c

    def forward(self, x):  # x: [P, N, C]
        P,N,C = x.shape
        x = self.mlp(x.reshape(-1,C)).reshape(P,N,-1)
        return x.max(1).values

class PointPillarsScatter(nn.Module):
    def __init__(self, C, grid_size):
        super().__init__()
        self.C, self.H, self.W = C, *grid_size

    def forward(self, feats, coords, batch_idx, B):
        imgs = torch.zeros(B,self.C,self.H,self.W, device=feats.device)
        imgs[batch_idx, :, coords[:,1], coords[:,0]] = feats
        return imgs

class PointPillars(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_points, max_pillars, num_ch=3):
        super().__init__()
        self.voxel_size, self.range = voxel_size, point_cloud_range
        self.max_points, self.max_pillars = max_points, max_pillars
        gx = int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])
        gy = int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])
        self.pfn = PillarFeatureNet(9,[num_ch*8,num_ch])
        self.pps = PointPillarsScatter(num_ch,(gx,gy))

    def forward(self, batch_points):
        pillars, coords, bidx = voxelize_batch(batch_points, self.range, self.voxel_size,
                                               self.max_points, self.max_pillars)
        feats = self.pfn(pillars)
        B = len(batch_points)
        return self.pps(feats, coords, bidx, B)


if __name__ == "__main__":

    # ----------------------------
    # Parameters
    # ----------------------------
    voxel_size = [0.625, 0.625, 0.1]   # [x, y, z] resolution
    point_cloud_range = [-40, -40, -1, 40, 40, 4]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    max_points_per_pillar = 300
    max_pillars = 2500
    device = "cuda:1"


    '''
    # ----------------------------
    # Load Pointcloud
    # ----------------------------
    file_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"
    pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    
    # Only keep x, y, z coordinates
    points = filter_pointcloud(pointcloud, point_cloud_range)
    

    pp = PointPillars(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_points= max_points_per_pillar, max_pillars=max_pillars, num_ch= 3)

    pseudo_img = pp(points)

    import matplotlib.pyplot as plt

    # Pick one channel to visualize
    channel0 = pseudo_img[0].cpu().detach().numpy()

    
    plt.scatter(pillar_coords[:,0].cpu(), pillar_coords[:,1].cpu(), s=5, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.title("Pillar coordinates")
    
    #channel0 = pseudo_img[0].cpu().numpy()
    plt.imshow(channel0, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("pseudo_image_channel0.png", dpi=300)
    plt.close()
    '''


    # THIS IS A BATCH EXAMPLE
    voxel_size = [0.625, 0.625, 0.1]
    pc_range = [-40,-40,-1,40,40,4]
    max_pts, max_pillars = 300, 2500
    device = 'cuda'

    pp = PointPillars(voxel_size, pc_range, max_pts, max_pillars).to(device)
    batch = [torch.rand(10000,4,device=device)*80-40 for _ in range(8)]  # 8 point clouds

    pseudo_imgs = pp(batch)  # [B, C, H, W]
    print(pseudo_imgs.shape)
    pseudo_img = pseudo_imgs[0,:,:,:]
    if pseudo_img.shape[0] == 3:
        # Normalize to [0,1] for display
        rgb_img = (pseudo_img - pseudo_img.min()) / (pseudo_img.max() - pseudo_img.min())
        rgb_img = rgb_img.permute(1,2,0)
        rgb_img = rgb_img.cpu().detach().numpy()

        import matplotlib.pyplot as plt
        plt.imshow(rgb_img)
        plt.title("Pseudo-image RGB (PCA)")
        plt.savefig("pseudo_image_rgb0.png", dpi=300)
        plt.close()

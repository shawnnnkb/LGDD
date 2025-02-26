import torch


def map2bev(cluster_outs, bev_size, batch_size, num_classes):

    cluster_feats = cluster_outs['cluster_feats']   # [N, C]
    cluster_xyz = cluster_outs['cluster_xyz']       # [N, 3]
    cluster_inds = cluster_outs['cluster_inds']     #[N, 3], (cls_id, batch_idx, cluster_id)

    ny = bev_size[0]
    nx = bev_size[1]
    in_channels = cluster_feats.shape[1]

    pixel_x = torch.div(cluster_xyz[:, 0] - 0, 0.16*2, rounding_mode='trunc')    # voxel_size=0.16, dowm_sampe_factor=2

    if num_classes == 4:
        pixel_y = torch.div((cluster_xyz[:, 1] - (-39.68)), 0.16*2, rounding_mode='trunc')    # for TJ4D
        indices_to_remove = ((pixel_x >= 248) | (pixel_y >= 216) | (pixel_x < 0) | (pixel_y < 0))
    elif num_classes == 3:
        pixel_y = torch.div(cluster_xyz[:, 1] - (-25.6), 0.16*2, rounding_mode='trunc')    # for VoD
        indices_to_remove = ((pixel_x >= 160) | (pixel_y >= 160) | (pixel_x < 0) | (pixel_y < 0))
    else:
        raise ValueError(f"Unsupported dataset, or check num_classes")

    pixel_x = pixel_x[~indices_to_remove]
    pixel_y = pixel_y[~indices_to_remove]
    cluster_feats = cluster_feats[~indices_to_remove]
    cluster_xyz = cluster_xyz[~indices_to_remove]
    cluster_inds = cluster_inds[~indices_to_remove]

    # print("pixel_x max: ", torch.max(pixel_x))
    # print("pixel_y max: ", torch.max(pixel_y))
    pixel = torch.stack((pixel_x, pixel_y), dim=1)

    if batch_size != 1:
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                in_channels,
                nx * ny,
                dtype=cluster_feats.dtype,
                device=cluster_feats.device)

            batch_mask = cluster_inds[:, 1] == batch_itt
            this_coors = pixel[batch_mask, :]
            indices = this_coors[:, 1] * ny + this_coors[:, 0]      # 顺序 ny 在前
            indices = indices.type(torch.long)
            voxels = cluster_feats[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            # 处理多个 pred_center 映射到同一个 pixel 的情况
            index_dict = {}  
            for i in range(len(indices)):  
                if indices[i].item() in index_dict:  
                    index_dict[indices[i].item()].append(i)
                else:  
                    index_dict[indices[i].item()] = [i]
            
            for key, value in index_dict.items():  
                if len(value) > 1: 
                    canvas[:, key] += torch.sum(voxels[:, value], dim=1)  
                else:
                    canvas[:, key] += voxels[:, value[0]] 

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, in_channels, ny, nx)

        return batch_canvas
        
    else:
        canvas = torch.zeros(
        in_channels,
        nx * ny,
        dtype=cluster_feats.dtype,
        device=cluster_feats.device)

        indices = pixel[:, 1] * ny + pixel[:, 0]
        indices = indices.long()
        voxels = cluster_feats.t()
        # Now scatter the blob back to the canvas.

        index_dict = {}  
        for i in range(len(indices)):  
            if indices[i].item() in index_dict:  
                index_dict[indices[i].item()].append(i) 
            else:  
                index_dict[indices[i].item()] = [i]
        
        for key, value in index_dict.items():  
            if len(value) > 1:
                canvas[:, key] += torch.sum(voxels[:, value], dim=1)  
            else:
                canvas[:, key] += voxels[:, value[0]]
        
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, in_channels, ny, nx)
        return canvas

def split_batch(cluster_outs, batch_size):
    """cluster_feats, (N, C) --> (B, M, C)
    
    """

    cluster_feats = cluster_outs['cluster_feats']   # [N, C]
    cluster_xyz = cluster_outs['cluster_xyz']       # [N, 3]
    cluster_inds = cluster_outs['cluster_inds'][:,1]     #[N, 3], (cls_id, batch_idx, cluster_id)

    N, C = cluster_feats.shape

    point_list = split_by_batch(cluster_feats, cluster_inds, batch_size)

    return point_list


def split_by_batch(data, batch_idx, batch_size):
    assert batch_idx.max().item() + 1 <= batch_size
    data_list = []
    for i in range(batch_size):
        sample_mask = batch_idx == i
        data_list.append(data[sample_mask])
    return data_list


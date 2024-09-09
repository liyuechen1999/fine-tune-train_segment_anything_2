# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def read_batch(data, class_id): # read random image and its annotaion from  the dataset (LabPics)

     #  select image

     ent  = data[np.random.randint(len(data))] # choose random entry
     gray_img = cv2.imread(ent["image"], cv2.IMREAD_GRAYSCALE)
     ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE) # read annotation]
     Img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

     # resize image
     r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) # scalling factor
     Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
     ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

     # Get binary masks and points
     inds = np.unique(ann_map)[1:] # load all indices
     points= []
     masks = []
     
     # 按类别遍历
     inds_class = np.intersect1d(class_id, np.unique(ann_map)[1:])
     for ind_class in inds_class:

          # 查找连通域
          ann_map = (ann_map == ind_class).astype(np.uint8)
          num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(ann_map)

          # 遍历连通域
          for ind in range(1, num_labels):
               mask=(labels_im == ind).astype(np.uint8) # make binary mask corresponding to index ind
               masks.append(mask)
               coords = np.argwhere(mask > 0) # get all coordinates in mask
               yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
               points.append([[yx[1], yx[0]]])
     
     return Img,np.array(masks),np.array(points), np.ones([len(masks),1])

def load_model(checkpoint_path, model_config, device='cuda:5'):
    """加载模型"""
    sam2_model = build_sam2(model_config, checkpoint_path, device=device) # load model
    return SAM2ImagePredictor(sam2_model)

# Set training parameters
def set_training_parameters(predictor):
    """设置训练参数"""
    predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
    predictor.model.image_encoder.train(True)

'''
#The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using:
predictor.model.image_encoder.train(True)
#Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
'''
def train_model(predictor, data, class_id, save_path, num_iterations=100000):
    """训练模型"""

    from datetime import datetime

    # 创建保存目录
    
    save_path = os.path.join(save_path,\
                             'class'+np.array2string(class_id).replace(' ','_'))

    # 判断目录是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler() # mixed precision
    best_iou = 0.1
    mean_iou = 0

    for itr in range(num_iterations):
        with torch.cuda.amp.autocast(): # cast to mix precision
            image, mask, input_point, input_label = read_batch(data, class_id) # load data batch
            if mask.shape[0] == 0: continue # ignore empty batches
            predictor.set_image(image) # apply SAM image encoder to the image

            # prompt encoding
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

            # mask decoder
            batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]) # Upscale the masks to the original image resolution

            # Segmentaion Loss calculation
            #gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            gt_mask = torch.tensor(mask.astype(np.float32)).to(predictor.device)  # 移动到指定设备
            prd_mask = torch.sigmoid(prd_masks[:, 0]) # Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05  # mix losses

            # apply back propagation
            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision

            # Display results
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            if best_iou < mean_iou:
                best_iou = mean_iou
                
                torch.save(predictor.model.state_dict(), os.path.join(save_path,"model.torch"))
                print("    save model in ", os.path.join(save_path,"model.torch"))
                print('    best iou: ', best_iou)

            print("step)", itr, "Accuracy(IOU)=", mean_iou)

def filter_data(data, class_id):
     # 过滤数据集，避免每次都循环
     filtered_data = []

     print('filter data, class:', class_id)
     for ent in data:
          gray_img = cv2.imread(ent["image"], cv2.IMREAD_GRAYSCALE)
          ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # read annotation

          # 判断是否存在该类别的标签
          #if gray_img is not None and ann_map is not None and np.isin(class_id, np.unique(ann_map)):
          if gray_img is not None and \
                 ann_map is not None and \
                 np.intersect1d(class_id, np.unique(ann_map)).size>0:
               
             filtered_data.append(ent)  # 仅添加符合条件的条目

     print('images num after filter: ',len(filtered_data))
     
     return filtered_data

def main():

     # Read data
     data_dir=r'/home/user/works/LLL-4090/segment-anything-2-main/data/road-dataset/' # Path to dataset
     data=[] # list of files in dataset
     #for ff, name in enumerate(os.listdir(data_dir+"images/")):  # go over all folder annotation
     image_path = os.path.join(data_dir, 'images/')
     for ff, name in enumerate(os.listdir(image_path)):  # go over all folder annotation
          data.append({"image":data_dir+"images/"+name,"annotation":data_dir+"labels-png/"+name[:-4]+".png"})

     # 单独微调灰度为2的类别
     class_id = np.array([1])

     save_path = '/home/user/works/LLL-4090/segment-anything-2-main/runs'

     # Load model
     sam2_checkpoint = "/home/user/works/LLL-4090/segment-anything-2-main/data/sam2_hiera_small.pt" # path to model weight
     model_cfg = "sam2_hiera_s.yaml" # model config
     predictor = load_model(sam2_checkpoint, model_cfg, device='cuda:1')

     # Set training parameters
     set_training_parameters(predictor)

     # 过滤数据集，仅保留有这个类别的数据
     fil_data = filter_data(data, class_id)
    
     # Start training
     print('Dataset path: ',data_dir)
     print('class_id: ', class_id)
     print('Start train!')
     train_model(predictor, fil_data, class_id, save_path)
     return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
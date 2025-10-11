import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from monai.transforms import LoadImaged, Compose, Orientationd
def getLargestCC(segmentation):
    labels = label(segmentation)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC

def test_all_case(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'/performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_all_case_LA(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4):
    with open('../data/LA/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["../data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case_Lung(model, image_list, num_classes=3, patch_size=(96, 96, 96), stride_xy=64, stride_z=64, save_result=False, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0, metric_txt_save=0):
    imagLoader = Compose([
        LoadImaged(keys=['image', 'label']),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=(('L', 'R'), ('P', 'A'), ('I', 'S')))
    ])
    ith = 0
    cls1_total_metric = 0.0
    cls2_total_metric = 0.0

    for image_path in image_list:
        imageName = image_path.name
        sample = {
            "image": image_path/(imageName + ".nii.gz"),
            "label": image_path/(imageName + "_label.nii.gz")
        }
        sample = imagLoader(sample)
        image = sample['image'].numpy()
        label = sample['label'].numpy()
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        results = calculate_metric_percase_multiclass(prediction, label, num_classes)
        
        cls1 = results[1]
        cls2 = results[2]

        if metric_detail:
            print(f"{imageName}: cls1_dice={cls1[0]:.4f},  cls2_dice={cls2[0]:.4f}")

        cls1_total_metric += np.asarray(cls1)
        cls2_total_metric += np.asarray(cls2)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    cls1_avg_metric = cls1_total_metric / len(image_list)
    cls2_avg_metric = cls2_total_metric / len(image_list)

    if metric_txt_save:
        print('Final Results:')
        print(f'cls1 average metric is {cls1_avg_metric}')
        print(f'cls2 average metric is {cls2_avg_metric}')
        with open(test_save_path+'/performance.txt', 'w') as f:
            f.writelines(f'cls1 average metric is {cls1_avg_metric} \n')
            f.writelines(f'cls2 average metric is {cls2_avg_metric} \n')

    return cls1_avg_metric, cls2_avg_metric

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

def calculate_metric_percase_multiclass(pred, gt, num_classes):
    if pred.shape != gt.shape:
        raise ValueError("Shape of pred and gt must be the same.")
    miss_pred_value=(0.0, 0.0, np.inf, np.inf)
    results = [[None, None, None, None]] * num_classes

    for c in range(1, num_classes):
        gt_c = (gt == c).astype(np.uint8)
        pred_c = (pred == c).astype(np.uint8)

        if gt_c.sum() == 0 and pred_c.sum() == 0:
            raise ValueError(f"Class {c} is missing in both prediction and ground truth.")  
        elif gt_c.sum() > 0 and pred_c.sum() == 0:
            print(f"Warning: Class {c} is missing in prediction.")
            dice, jc, hd, asd = miss_pred_value
        else:
            dice = metric.binary.dc(pred_c, gt_c)
            jc = metric.binary.jc(pred_c, gt_c)
            hd = metric.binary.hd95(pred_c, gt_c)
            asd = metric.binary.asd(pred_c, gt_c)

        results[c] = ([
            float(dice) if np.isfinite(dice) else float(np.inf),
            float(jc) if np.isfinite(jc) else float(np.inf),
            float(hd) if np.isfinite(hd) else float(np.inf),
            float(asd) if np.isfinite(asd) else float(np.inf)
        ])

    return np.asarray(results)


# coding=utf-8
from __future__ import print_function, division
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from utils.shap_utils import *
import os
from torchvision import models, transforms
from sam_explainer import *
from tqdm import tqdm

# 合并两个路径
path_join = lambda root, leaf: os.path.join(root, leaf)
# 将root路径下含有字段string的所有文件的路径生成一个列表
select_path_list = lambda root, string: [path_join(root, label) for label in os.listdir(root) if string in label]
# 检查path文件夹是否存在 如果不存在则创建一个
check_path_or_create = lambda path: os.makedirs(path) if not os.path.exists(path) else None


# ImageNet文件的路径
imagenet_path = os.path.join('..', '..', 'dataset', 'ds004496', 'stimuli', 'imagenet')
dirs = [os.path.join(imagenet_path, d) for d in os.listdir(imagenet_path) if not d == 'info']

### define a tagret model
model = models.resnet50(weights='IMAGENET1K_V2').cuda()
cvmodel = model.cuda()
cvmodel.eval()
feat_exp = create_feature_extractor(cvmodel, return_nodes=['avgpool'])
fc = model.fc
model.eval()
feat_exp.eval()
cvmodel.eval()


### define a sam
sam = sam_model_registry["default"](checkpoint=os.path.join('..', 'meta_sam_checkpoints', 'sam_vit_h_4b8939.pth'))
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)


### define three wats to pre-process a given image
test_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) ### a complete imagenet data pre-process

image_reshape = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
]) ### process the imagenet data to sam


image_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) ### imagenet std and mean

def run_samshap(input_path : str, output_path : str):
    img = Image.open(input_path).convert('RGB')

    predict_org = torch.nn.functional.softmax(model(test_preprocess(img).unsqueeze(0).cuda()),dim=1)
    pred_image_class = int(torch.argmax(predict_org))

    for_mask_image = np.array(image_reshape(img)) ### np int type

    input_image_copy = for_mask_image.copy()
    org_masks = gen_concept_masks(mask_generator,input_image_copy)
    concept_masks = np.array([i['segmentation'].tolist() for i in org_masks])
    auc_mask, shap_list = samshap(model,input_image_copy,pred_image_class,concept_masks,fc,feat_exp,image_norm=image_norm)

    ### select the concept patch with the highest shapley value
    final_explain = (for_mask_image*auc_mask[0])

    final_explain = Image.fromarray(final_explain)
    final_explain.save(output_path)

samshap_output_path = os.path.join('..', '..', 'samshap_output')
imagenet_output_path = os.path.join(samshap_output_path, 'imagenet_output')
check_path_or_create(imagenet_output_path)

for dir in dirs:
    save_path = os.path.join(imagenet_output_path, dir)
    check_path_or_create(save_path)
    for  file, _ in zip(os.listdir(dir), tqdm(range(len(os.listdir(dir))))):
        source_file = os.path.join(dir, file)
        dest_file = os.path.join(save_path, file)
        run_samshap(input_path=source_file, output_path=dest_file)
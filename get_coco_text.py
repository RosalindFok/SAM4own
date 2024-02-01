# coding=utf-8
import os
import json
import rich.progress

mscoco_path = os.path.join('..', 'dataset', 'MScoco')

# 合并两个路径
path_join = lambda root, leaf: os.path.join(root, leaf)
# 将root路径下含有字段string的所有文件的路径生成一个列表
select_path_list = lambda root, string: [path_join(root, label) for label in os.listdir(root) if string in label]

dirs = [path_join(mscoco_path, d) for d in os.listdir(mscoco_path) if os.path.isdir(path_join(mscoco_path, d))]

# 将多维嵌套列表展平成一维列表
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            # 如果当前项是列表，则递归调用flatten_list
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

annotations_path_list = flatten_list([select_path_list(dir, 'annotations') for dir in dirs])

def read_json(file : str)->dict:
    with rich.progress.open(file, 'r') as f:
        return json.load(f)

def save_json(file : str, content : dict)->None:
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4, ensure_ascii=False)

for annotations_path in annotations_path_list:
    for file in [x for x in os.listdir(annotations_path) if os.path.isfile(path_join(annotations_path, x))]:
        """ 需要的文件
        2014
        captions_train2014.json             dict_keys(['info', 'images', 'licenses', 'annotations'])
        captions_val2014.json               dict_keys(['info', 'images', 'licenses', 'annotations'])
        ------------------------------------------------------
        2017
        captions_train2017.json             dict_keys(['info', 'licenses', 'images', 'annotations'])
        captions_val2017.json               dict_keys(['info', 'licenses', 'images', 'annotations'])
        """
        # 过滤掉不需要的文件
        if not 'captions_' in file:
            continue

        data = read_json(path_join(annotations_path, file))
        
        save_dict = {} # {file_name : {height:height, width:width, captions:captions}}

        images_dict, annotations_dict = {}, {}
        for  img in data['images']:
            images_dict[img['id']] = {'file_name':img['file_name'], 'height': img['height'], 'width': img['width']}
        for ann in data['annotations']:
            annotations_dict[ann['image_id']] = ann['caption']
        assert len(images_dict) == len(annotations_dict)

        for key, value in images_dict.items():
            save_dict[value['file_name']] = {'height':value['height'], 'width':value['width'], 'captions':annotations_dict[key]}
        
        save_path = os.path.join(mscoco_path, 'simplify'+'_'+file)
        save_json(file=save_path, content=save_dict)
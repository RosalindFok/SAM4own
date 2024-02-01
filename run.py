import os

# 合并两个路径
path_join = lambda root, leaf: os.path.join(root, leaf)
# 将root路径下含有字段string的所有文件的路径生成一个列表
select_path_list = lambda root, string: [path_join(root, label) for label in os.listdir(root) if string in label]
# 检查path文件夹是否存在 如果不存在则创建一个
check_path_or_create = lambda path: os.makedirs(path) if not os.path.exists(path) else None

# 检查数据集是否存在
dataset_path = path_join('..', 'dataset')
algnauts_2023_challenge_data_path = path_join(dataset_path, 'algnauts_2023_challenge_data')
ds004496_path = path_join(dataset_path, 'ds004496')
for path in [dataset_path, algnauts_2023_challenge_data_path, ds004496_path]:
    if not os.path.exists(path): 
        print(f'Error: {path} does not exist. Please Check.')

# segment-anything模型分割的结果
sam_output_result_path = path_join('..', 'sam_output')
algnauts_result_path = path_join(sam_output_result_path, 'algnauts_result')
ds004496_result_path = path_join(sam_output_result_path, 'ds004496_result')
check_path_or_create(sam_output_result_path)
check_path_or_create(algnauts_result_path)
check_path_or_create(ds004496_result_path)

# 使用SAM进行语义分割
def sam(input_path : str, output_path : str, min_mask_region_area : int = 10):
    check_path_or_create(output_path)
    # 使用huge模型
    huge_ckpt =  path_join('meta_sam_checkpoints', 'sam_vit_h_4b8939.pth')
    amg_py_path = os.path.join('segment-anything', 'scripts', 'amg.py')
    os.system(f'python {amg_py_path} --checkpoint {huge_ckpt} --model-type vit_h --input {input_path} --output {output_path} --min-mask-region-area {min_mask_region_area}')



# 加载输入的自然图像的路径
for path in [algnauts_2023_challenge_data_path, ds004496_path]:
    # algnauts_2023_challenge_data
    if path == algnauts_2023_challenge_data_path:
        subjs_dir_path = [path_join(algnauts_2023_challenge_data_path, d) for d in os.listdir(path) if os.path.isdir(path_join(path, d))]
        for subj_dir_path in subjs_dir_path:
            subj_id = subj_dir_path.split(os.sep)[-1]
            test_images_dir_path = os.path.join(subj_dir_path, 'test_split', 'test_images')
            training_images_dir_path = os.path.join(subj_dir_path, 'training_split', 'training_images')
            for input_path in [test_images_dir_path, training_images_dir_path]:
                # test
                if 'test' in input_path:
                    sam(input_path=input_path, output_path=os.path.join(algnauts_result_path, subj_id, 'test_result'))
                # training
                elif 'training' in input_path:
                    sam(input_path=input_path, output_path=os.path.join(algnauts_result_path, subj_id, 'training_result'))

    # ds004496
    elif path == ds004496_path:
        stimuli_dir_path = path_join(path, 'stimuli')
        coco_dir_path = path_join(stimuli_dir_path, 'coco') # 此外还有floc/imagenet/prf
        sam(coco_dir_path, path_join(ds004496_result_path, 'coco'))
        # imagenet_dir_path = path_join(stimuli_dir_path, 'imagenet')

# Cleanup unnecessary files and optimize the local repository
os.system('git gc --prune=now')
import os

input_path = os.path.join('..', 'input')
output_path = os.path.join('..', 'output')
for path in [input_path, output_path]:
    if not os.path.exists(path): 
        os.makedirs(path)  

amg_py_path = os.path.join('segment-anything', 'scripts', 'amg.py')

# 使用huge模型
huge_ckpt =  os.path.join('meta_sam_checkpoints', 'sam_vit_h_4b8939.pth')
os.system(f'python {amg_py_path} --checkpoint {huge_ckpt} --model-type vit_h --input {input_path} --output {output_path}')
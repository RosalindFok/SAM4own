# SAM4own
Segment-Anything model for my own use

## 文件夹meta_sam_checkpoints
sam_vit_b_01ec64.pth：b表示base(最小) <br>
sam_vit_h_4b8939.pth：h表示huge(最大) <br>
sam_vit_l_0b3195.pth：l表示large 

## 文件夹segment-anything
克隆仓库：`git clone https://github.com/facebookresearch/segment-anything`    <br>
安装项目(-e后的英文句点需要输入的)：`cd segment-anything; pip install -e .`     

## 文件夹samshap-hkust
克隆仓库: `git clone https://github.com/Jerry00917/samshap.git`    <br>

## 配置环境
`conda create --name SAM python=3.9`    <br>
`source activate SAM`    <br>
`pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/`     <br>
`pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/`     <br>
`pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple/`     <br>

`pip install torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple/`     <br>

## 下载模型
`cd` into `SAM4own/meta_sam_checkpoints`   <br>
`sh run_download_checkpoints.sh`

## 运行脚本 - Meta的SAM
`cd` into `SAM4own`   <br>
`python run.py`

## 运行脚本 - HKUST的SAMSHAT
`cd` into `samshap-hkust`   <br>
`python demo_samshap.py`

### Note
- SAMSHAT使用了ResNet50的权重文件，在`demo_samshap.py`可对`models.resnet50(weights='IMAGENET1K_V2')`进行更换修改
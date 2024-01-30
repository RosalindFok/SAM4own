# SAM4own
Segment-Anything model for my own use
## 文件夹meta_sam_checkpoints
sam_vit_b_01ec64.pth：b表示base(最小) <br>
sam_vit_h_4b8939.pth：h表示huge(最大) <br>
sam_vit_l_0b3195.pth：l表示large 

## 文件夹segment-anything
克隆仓库：`git clone https://github.com/facebookresearch/segment-anything`    <br>
安装项目(-e后的英文句点需要输入的)：`cd segment-anything; pip install -e .`     

## 配置环境
`conda create --name SAM python=3.9`    <br>
`source activate SAM`    <br>
`pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/` 

## 下载模型
`cd` into `SAM4own/meta_sam_checkpoints`   <br>
`sh run_download_checkpoints.sh`

## 运行脚本
`cd` into `SAM4own`   <br>
`python run.py`

# com.mcast.res.yolov7
Configuration for Yolov7

# Setting up
- Download and install latest Nvidia Drivers

**P.S.** Restart PC after this step

- Download and install [cuda](https://developer.nvidia.com/cuda-downloads)

**P.S.** Restart PC after this step

- Download and install VS Studio Community, install C++ developer package
- Download and install Anaconda/Miniconda
- Download or clone Yolo [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- Download pretrained weights from repository page.

**P.S.** Yolo should be in a git repository

- In command prompt check cuda version by running `nvcc -V`
- Go to [PyTorch](https://pytorch.org/get-started/locally/) to get the command for your compatible version. Revise conda environment command below.
- Create conda environment

```
conda create --name yolo python=3.10
conda activate yolo
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Consider revising tensorboard version to <2.11 due to lack of native GPU support on Windows
pip install -r requirements.txt
pip install "tensorflow<2.11"
conda install jupyter
ipython kernel install --name "yolo-kernel" --user
conda install -c conda-forge nomkl
conda clean --all
```

- Test with following and check output in `runs` folder:

`python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg --device 0`

# Troubleshooting
- For memory issues open train.py in yolov7 and add the following code after line 479  as per https://github.com/WongKinYiu/yolov7/issues/735

```
torch.cuda.empty_cache()
gc.collect()
```

- If getting queue empty error during train command set `--workers 0`


# com.mcast.res.yolov7
Configuration for Yolov7

# Setting up
- Download and install latest [Nvidia Video Driver](https://www.nvidia.com/download/index.aspx)

**P.S.** Restart PC after this step

- Download and install [cuda](https://developer.nvidia.com/cuda-downloads)

**P.S.** Restart PC after this step

- Download and install [Visual C++ via Microsoft Visual Studio Community](https://visualstudio.microsoft.com/vs/community/), install C++ developer package
- Download and install [Anaconda](https://anaconda.org/)/[MiniConda](https://docs.conda.io/en/latest/miniconda.html)
- Download or clone [Yolo](https://github.com/WongKinYiu/yolov7)
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
conda install -c conda-forge wandb
conda install -c conda-forge nomkl # This is due to an error documented below. It will find conflicst and take long. Consider without first.
conda clean --all
```

- Test with following and check output in `runs` folder:

`python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg --device 0`

# Troubleshooting
- [Memory Issues](https://github.com/WongKinYiu/yolov7/issues/735): For memory issues open train.py in yolov7 and add the following code after line 479:

```
torch.cuda.empty_cache()
gc.collect()
```

- [Empty Queue](https://github.com/ultralytics/yolov5/issues/1675): If getting queue empty error during train command, set `--workers 0`

- [libiomp5md.dll](https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a): If getting MP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. There are two solutions: first installation of nomkl which should have already been done; secondly adding

```
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

# Useful links
- [wandb](https://wandb.ai/)

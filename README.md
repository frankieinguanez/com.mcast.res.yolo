# com.mcast.res.yolov7
Configuration for Yolov7

# Setting up on Windows 11 with native GPU support.
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
- Open an Anaconda Prompt as administrator and create conda environment

```
conda create --name yolo python=3.10
conda activate yolo
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Consider revising tensorboard version to <2.11 due to lack of native GPU support on Windows
pip install -r requirements.txt
pip install "tensorflow<2.11"
conda install -c conda-forge wandb # This would require further setting up, check useful links below
conda clean --all
```

- To verify if PyTorch is using GPU type the following command that utilize [torch package](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) in the Anaconda prompt with the respective conda environment activated:

```
python

import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
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

- [libiomp5md.dll](https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a): If getting MP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized add the following in train.py

```
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

- [Resuming training](https://github.com/ultralytics/yolov5/issues/911) and addressing [corrupt weights file](https://github.com/pytorch/pytorch/issues/31620): To resume training of a model should it stop for some reason use `python train.py --resume`. If a custom data option was provided then provide it again. If the following error occurs `PytorchStreamReader failed reading zip archive: failed finding central directory (no backtrace available)` than the last checkpoint is corrupted. Navigate to the weights folder in the run folder, delete `last.pt`, take the most recent epoch weights file, copy it and rename it to `last.pt`.

# Useful links
- [wandb](https://wandb.ai/)
- [Deep dive in Yolov7](https://towardsdatascience.com/yolov7-a-deep-dive-into-the-current-state-of-the-art-for-object-detection-ce3ffedeeaeb)
- [Google Remote Desktop](https://remotedesktop.google.com/)

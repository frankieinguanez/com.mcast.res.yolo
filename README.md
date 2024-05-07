# com.mcast.res.yolo
Configuration for different YOLO

# Setting up on Windows 11 with native GPU support.
- Download and install latest [Nvidia Video Driver](https://www.nvidia.com/download/index.aspx)

**P.S.** Restart PC after this step

- Download and install [cuda](https://developer.nvidia.com/cuda-downloads)

**P.S.** Restart PC after this step

- Download and install [Visual C++ via Microsoft Visual Studio Community](https://visualstudio.microsoft.com/vs/community/), install C++ developer package
- Download pretrained weights from repository page.

**P.S.** Yolo should be in a git repository

- In command prompt check cuda version by running `nvcc -V`
- Go to [PyTorch](https://pytorch.org/get-started/locally/) to get the command for your compatible version. 
- Open a command prompt where you want to create the Python Virtual environment

```
python -m venv yolo_v8
.\yolo_v8\Scripts\activate
pip install ultralytics
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.\yolo_v8\Scripts\deactivate
```

- To verify native GPU support run ```test_pu.py```. To test PyTorch run ```test_cuda.py```.


# Useful links
- [wandb](https://wandb.ai/)
- [Deep dive in Yolov7](https://towardsdatascience.com/yolov7-a-deep-dive-into-the-current-state-of-the-art-for-object-detection-ce3ffedeeaeb)
- [Google Remote Desktop](https://remotedesktop.google.com/)

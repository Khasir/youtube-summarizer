# YouTube Summarizer
Summarize YouTube videos

## How it works
1. Input how many days' worth of videos (N) you want to summarize.
2. Input which channel C to summarize.
3. Download N days of video audio from channel C.
4. Detect language and perform text-to-speech on those videos.
5. Generate multiple summaries:
- A summary for each video
- An overall summary for all videos.

## Setup
### Environment
```sh
mkdir .venv
python -m venv ./.venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install yt-dlp==2024.5.27 flask==3.0.3
pip install transformers==4.41.2 #accelerate==0.31.0
```

Check which version of CUDA you have (Nvidia GPU):
```sh
> nvidia-smi
Sat Jun 22 15:35:33 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.99                 Driver Version: 555.99         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060      WDDM  |   00000000:06:00.0  On |                  N/A |
|  0%   50C    P8             13W /  170W |     984MiB /  12288MiB |      7%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

However, `nvcc --version` fails for me and is not recognized. This indicates that the CUDA 12.5 above is not actually accessible.

If you need to swap CUDA versions, follow this [guide](https://github.com/bycloudai/SwapCudaVersionWindows) or the [official documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#installing-cuda-development-tools). The guide seems to assume that you have an alternate version already installed.

I will be installing [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11) since it's the latest supported stable version for PyTorch as of June 2024.

Install [PyTorch](https://pytorch.org/) accordingly. For example:
```sh
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

### YouTube API

Follow the setup here to enable access to YouTube's API: https://developers.google.com/youtube/v3/getting-started

Then install the required [client library](https://developers.google.com/youtube/v3/quickstart/python) (we can skip the autorization libraries since we're accessing public information):
```sh
pip install --upgrade google-api-python-client
```


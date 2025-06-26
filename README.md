# Pay Less Attention to Deceptive Artifacts (PLADA)
<div align=center>
<img width="350" alt="1723450312316" src="https://github.com/user-attachments/assets/83839e33-37a3-4cd0-a6cc-dc524e2cb5a7">  
</div>

>  ❗️❗️❗️ **News:**
> 
> ✨2025.6.25: We release code of PLADA, a new deepfake detection framework addressing challenges in online social networks. You can find our work in [arXiv](https://arxiv.org/abs/2506.20548) now !

## TODO List
> ✔️ ~~Upload Traning Code~~
> 
> ✔️ ~~Upload Inference Code~~
> 
> ⏱️ Pretrained Weight and Preprocessed Datasets (Once our paper is accepted)

## ⏳ Quick Start

### 1. Installation
You can run the following script to configure the necessary environment:

```
git clone https://github.com/ManyiLee/PLADA.git
cd PLADA
conda create -n plada python=3.9
conda activate plada
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

### 2. Download Data

<a href="#top">[Back to top]</a>

⭐️ **Datasets** (26 widely used datasets):
> **9 GANS**: InfoGAN、BEGAN、CramGAN、AttGAN、MMDGAN、RelGAN、S3GAN、SNGGAN、STGGAN
>
> **8 GANs**: ProGAN、StyleGAN、StyleGAN2、BigGAN、CycleGAN、StarGAN、GuaGAN、Deepfake
>
> **5 Diffusions**: DALLE、guided-diffusion、improved-diffusion、midjourney、ddpm-google tan
>
> **8 Diffusions**: dalle、glide_100_10、glide_100_27、glide_50_27、guided、ldm_100、ldm_200、ldm_200_cfg

Detailed information about the datasets used in PLADA is summarized below:

| Dataset | Function | Original Repository |
| --- | --- | --- |
| ForenSynths(ProGAN) | Train | [Hyper-link](https://github.com/PeterWang512/CNNDetection) |
| 8GANs | Test | [Hyper-link](https://github.com/PeterWang512/CNNDetection) |
| 9GANs | Test | [Hyper-link](https://github.com/chuangchuangtan/GANGen-Detection) |
| 5 Diffusions | Test | [Hyper-link](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) |
| 8 Diffusions | Test | [Hyper-link](https://github.com/Yuheng-Li/UniversalFakeDetect) |


Upon downloading the datasets, please ensure to store them in the [`./datasets`](./datasets/) folder, arranging them in accordance with the directory structure outlined below:

```
datasets
├── train
|   ├── NoCcmp
|   |   ├── airplane
|   |   ├── bicycel
|   |   ├── ......
|   ├── 20%StaticCmp
|   ├── ......
|   ├── other Ratio
├── test
|   ├── AttGAN
|   |   ├── NoCmp
|   │   │   ├──0_real
|   │   │   ├──1_fake
|   |   ├── RandomCmp
|   |   ├── StactioCmp
|   ├── BEGAN
|   ├── ......
├── val
|   ├── Nocmp
|   |   ├── airplane
|   |   ├── bicycle
|   |   ├── ......
|   ├── RandomCmp
|   ├── StaticCmp
```

### 3. Preprocessing
You can run the following script to preprocess images as our experimental setting:

```
python preprocess/random_compression.py -r 1.0 -d 9Gans -m RandomCmp -up 100 -down 30 -t test
python preprocess/random_compression.py -r 1.0 -d 8Gans -m RandomCmp -up 100 -down 30 -t test
python preprocess/random_compression.py -r 1.0 -d 5Diffusions -m RandomCmp -up 100 -down 30 -t test
python preprocess/random_compression.py -r 1.0 -d 8Diffusions -m RandomCmp -up 100 -down 30 -t test
python preprocess/random_compression.py -r 1.0 -d 9Gans -m StaticCmp -up 50 -down 50 -t test
python preprocess/random_compression.py -r 1.0 -d 8Gans -m StaticCmp -up 50 -down 50 -t test
python preprocess/random_compression.py -r 1.0 -d 5Diffusions -m StaticCmp -up 50 -down 50 -t test
python preprocess/random_compression.py -r 1.0 -d 8Diffusions -m StaticCmp -up 50 -down 50 -t test
python preprocess/random_compression.py -r 0.2 -d ProGan -m RandomCmp -up 100 -down 30 -t train
python preprocess/random_compression.py -r 0.2 -d ProGan -m StaticCmp -up 50 -down 50 -t test
```
And if you want to try other comfigurations, please adjust the arguments.

### 4. Replace File Paths

Please replace dataset path in below files:
```
./configs/run.yaml
./eval/eval_dataset.py
./preprocess/random_compression.py
./util.py
```

### 5. Training

<a href="#top">[Back to top]</a>

```
python train.py -g 0,1
```

You can also adjust the training and testing argument by modifying the config file. By default, the checkpoints and features will be saved during the training process.


## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our work useful to your research, please cite it as follows:

```
@misc{li2025payattentiondeceptiveartifacts,
      title={Pay Less Attention to Deceptive Artifacts: Robust Detection of Compressed Deepfakes on Online Social Networks}, 
      author={Manyi Li and Renshuai Tao and Yufan Liu and Chuangchuang Tan and Haotong Qin and Bing Li and Yunchao Wei and Yao Zhao},
      year={2025},
      eprint={2506.20548},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.20548}, 
}
```

# 🦙 LaMa：基于傅里叶卷积的分辨率鲁棒大掩码图像修复

作者：Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin,
Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, Victor Lempitsky.

<p align="center" "font-size:30px;">
  🔥🔥🔥
  <br>
  <b>
LaMa 能够出色地泛化到远高于训练分辨率（256x256）的更高分辨率（约2k❗️），即使在具有挑战性的场景中（如周期性结构的补全）也能取得优秀的效果。</b>
</p>

[[项目主页](https://advimman.github.io/lama-project/)] [[arXiv](https://arxiv.org/abs/2109.07161)] [[补充材料](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf)] [[BibTeX](https://senya-ashukha.github.io/projects/lama_21/paper.txt)] [[Casual GAN Papers 摘要](https://www.casualganpapers.com/large-masks-fourier-convolutions-inpainting/LaMa-explained.html)]

<p align="center">
  <a href="https://colab.research.google.com/drive/15KTEIScUbVZtUP6w2tCDMVpE-b1r9pkZ?usp=drive_link">
  <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
      <br>
   在 Google Colab 中试用
  <br>
  所有 Yandex 下载链接已失效，您可以从以下地址下载模型：https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=sharing
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/ezgif-4-0db51df695a8.gif" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/gif_for_lightning_v1_white.gif" />
</p>



# LaMa 相关开发
（欢迎通过创建 issue 分享您的论文）
- https://github.com/geekyutao/Inpaint-Anything --- Inpaint Anything：Segment Anything 与图像修复的结合
<p align="center">
  <img src="https://raw.githubusercontent.com/geekyutao/Inpaint-Anything/main/example/MainFramework.png" />
</p>

- [特征细化以改进高分辨率图像修复](https://arxiv.org/abs/2206.13644) / [视频](https://www.youtube.com/watch?v=gEukhOheWgE) / 代码 https://github.com/advimman/lama/pull/112 / 由 Geomagical Labs 提供（[geomagical.com](geomagical.com)）
<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/images/FeatureRefinement.png" />
</p>

# 非官方第三方应用：
（欢迎通过创建 issue 分享您的应用/实现/演示）

- https://github.com/enesmsahin/simple-lama-inpainting - 一个简单的 LaMa 修复 pip 包。
- https://github.com/mallman/CoreMLaMa - Apple Core ML 模型格式
- [https://cleanup.pictures](https://cleanup.pictures/) - 一个简单的交互式物体移除工具，由 [@cyrildiagne](https://twitter.com/cyrildiagne) 开发
    - [lama-cleaner](https://github.com/Sanster/lama-cleaner)，由 [@Sanster](https://github.com/Sanster/lama-cleaner) 开发，是 [https://cleanup.pictures](https://cleanup.pictures/) 的自托管版本
- 已集成到 [Huggingface Spaces](https://huggingface.co/spaces)，使用 [Gradio](https://github.com/gradio-app/gradio)。查看演示：[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/lama)，由 [@AK391](https://github.com/AK391) 提供
- Telegram 机器人 [@MagicEraserBot](https://t.me/MagicEraserBot)，由 [@Moldoteck](https://github.com/Moldoteck) 开发，[代码](https://github.com/Moldoteck/MagicEraser)
- [Auto-LaMa](https://github.com/andy971022/auto-lama) = DE:TR 目标检测 + LaMa 修复，由 [@andy971022](https://github.com/andy971022) 开发
- [LAMA-Magic-Eraser-Local](https://github.com/zhaoyun0071/LAMA-Magic-Eraser-Local) = 基于 PyQt5 构建的独立修复应用程序，由 [@zhaoyun0071](https://github.com/zhaoyun0071) 开发
- [Hama](https://www.hama.app/) - 使用智能画笔简化掩码绘制的物体移除工具。
- [ModelScope](https://www.modelscope.cn/models/damo/cv_fft_inpainting_lama/summary) = 中文最大的模型社区，由 [@chenbinghui1](https://github.com/chenbinghui1) 提供。
- [LaMa with MaskDINO](https://github.com/qwopqwop200/lama-with-maskdino) = MaskDINO 目标检测 + LaMa 修复（含细化），由 [@qwopqwop200](https://github.com/qwopqwop200) 开发。
- [CoreMLaMa](https://github.com/mallman/CoreMLaMa) - 将 Lama Cleaner 移植的 LaMa 转换为 Apple Core ML 模型格式的脚本。

# 环境配置

❗️❗️❗️ 所有 Yandex 下载链接已失效，您可以从 [Google Drive](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=sharing) 下载模型 ❗️❗️❗️

克隆仓库：
`git clone https://github.com/advimman/lama.git`

有三种环境配置方式：

1. Python virtualenv：

    ```
    virtualenv inpenv --python=/usr/bin/python3
    source inpenv/bin/activate
    pip install torch==1.8.0 torchvision==0.9.0

    cd lama
    pip install -r requirements.txt
    ```

2. Conda

    ```
    % 在 Linux 上安装 conda，其他操作系统请在 https://docs.conda.io/en/latest/miniconda.html 下载 miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    $HOME/miniconda/bin/conda init bash

    cd lama
    conda env create -f conda_env.yml
    conda activate lama
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
    pip install pytorch-lightning==1.2.9
    ```

3. Docker：无需任何操作 🎉。

# 推理 <a name="prediction"></a>

运行：
```
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

**1. 下载预训练模型**

最佳模型（Places2, Places Challenge）：

```
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
```

所有模型（Places 和 CelebA-HQ）：

```
从 [https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=drive_link] 下载
unzip lama-models.zip
```

**2. 准备图像和掩码**

下载测试图像：

```
unzip LaMa_test_images.zip
```
<details>
 <summary>或者自行准备数据：</summary>
1) 创建命名格式为 `[图片名]_maskXXX[图片后缀]` 的掩码，并将图像和掩码放在同一个文件夹中。

- 您可以使用此[脚本](https://github.com/advimman/lama/blob/main/bin/gen_mask_dataset.py)进行随机掩码生成。
- 检查文件格式：
    ```
    image1_mask001.png
    image1.png
    image2_mask001.png
    image2.png
    ```

2) 在 `configs/prediction/default.yaml` 中指定 `image_suffix`，例如 `.png` 或 `.jpg` 或 `_input.jpg`。

</details>


**3. 预测**

在主机上：

    python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

**或者**在 Docker 中

以下命令将从 Docker Hub 拉取 Docker 镜像并执行预测脚本：
```
bash docker/2_predict.sh $(pwd)/big-lama $(pwd)/LaMa_test_images $(pwd)/output device=cpu
```
Docker CUDA：
```
bash docker/2_predict_with_gpu.sh $(pwd)/big-lama $(pwd)/LaMa_test_images $(pwd)/output
```

**4. 带细化的预测**

在主机上：

    python3 bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

# 训练与评估

确保先运行：

```
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

然后下载 _感知损失_ 所需的模型：

    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth


## Places

⚠️ 注意：LaMa 论文中 Places 数据集的 FID/SSIM/LPIPS 指标值是在下面评估部分生成的 30000 张图像上计算得到的。
有关评估数据的更多详情，请查看 [[补充材料第 3 节：数据集划分](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf#subsection.3.1)]  ⚠️

在主机上：

    # 从 http://places2.csail.mit.edu/download.html 下载数据
    # Places365-Standard：从高分辨率图像部分下载 Train(105GB)/Test(19GB)/Val(2.1GB)
    wget http://data.csail.mit.edu/places/places365/train_large_places365standard.tar
    wget http://data.csail.mit.edu/places/places365/val_large.tar
    wget http://data.csail.mit.edu/places/places365/test_large.tar

    # 解压训练/测试/验证数据并创建 .yaml 配置文件
    bash fetch_data/places_standard_train_prepare.sh
    bash fetch_data/places_standard_test_val_prepare.sh

    # 为每个 epoch 结束时的测试和可视化采样图像
    bash fetch_data/places_standard_test_val_sample.sh
    bash fetch_data/places_standard_test_val_gen_masks.sh

    # 运行训练
    python3 bin/train.py -cn lama-fourier location=places_standard

    # 为了评估训练好的模型并报告与论文一致的指标，
    # 我们需要采样之前未见过的 3 万张图像并为它们生成掩码
    bash fetch_data/places_standard_evaluation_prepare_data.sh

    # 在 256 和 512 的粗/细/中等掩码上进行模型推理并运行评估
    # 示例如下：
    python3 bin/predict.py \
    model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier_/ \
    indir=$(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
    outdir=$(pwd)/inference/random_thick_512 model.checkpoint=last.ckpt

    python3 bin/evaluate_predicts.py \
    $(pwd)/configs/eval2_gpu.yaml \
    $(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
    $(pwd)/inference/random_thick_512 \
    $(pwd)/inference/random_thick_512_metrics.csv



Docker：待完成

## CelebA
在主机上：

    # 确保您在 lama 文件夹中
    cd lama
    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    # 下载 CelebA-HQ 数据集
    # 从 https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P 下载 data256x256.zip

    # 解压并拆分为训练/测试/可视化集 & 创建配置文件
    bash fetch_data/celebahq_dataset_prepare.sh

    # 为每个 epoch 结束时的测试和可视化测试生成掩码
    bash fetch_data/celebahq_gen_masks.sh

    # 运行训练
    python3 bin/train.py -cn lama-fourier-celeba data.batch_size=10

    # 在 256 的粗/细/中等掩码上进行模型推理并运行评估
    # 示例如下：
    python3 bin/predict.py \
    model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier-celeba_/ \
    indir=$(pwd)/celeba-hq-dataset/visual_test_256/random_thick_256/ \
    outdir=$(pwd)/inference/celeba_random_thick_256 model.checkpoint=last.ckpt


Docker：待完成

## Places Challenge

在主机上：

    # 此脚本并行下载多个 .tar 文件并解压
    # Places365-Challenge：从高分辨率图像下载 Train(476GB)（用于训练 Big-Lama）
    bash places_challenge_train_download.sh

    待完成：准备
    待完成：训练
    待完成：评估

Docker：待完成

## 创建自定义数据

请查看 CelebAHQ 部分的数据准备和掩码生成 bash 脚本，
如果您在以下某个步骤中遇到困难。


在主机上：

    # 确保您在 lama 文件夹中
    cd lama
    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    # 您需要准备以下图像文件夹：
    $ ls my_dataset
    train
    val_source # 2000 张或更多图像
    visual_test_source # 100 张或更多图像
    eval_source # 2000 张或更多图像

    # LaMa 在训练时动态生成随机掩码，
    # 但测试和可视化测试需要固定的掩码以确保评估一致性。

    # 假设我们想在 512x512 的验证数据集上使用粗/细/中等掩码评估和选择最佳模型
    # 并且您的图像扩展名为 .jpg：

    python3 bin/gen_mask_dataset.py \
    $(pwd)/configs/data_gen/random_<size>_512.yaml \ # thick, thin, medium
    my_dataset/val_source/ \
    my_dataset/val/random_<size>_512.yaml \# thick, thin, medium
    --ext jpg

    # 掩码生成器将：
    # 1. 调整大小并裁剪验证图像，保存为 .png
    # 2. 生成掩码

    ls my_dataset/val/random_medium_512/
    image1_crop000_mask000.png
    image1_crop000.png
    image2_crop000_mask000.png
    image2_crop000.png
    ...

    # 为 visual_test 文件夹生成粗/细/中等掩码：

    python3 bin/gen_mask_dataset.py \
    $(pwd)/configs/data_gen/random_<size>_512.yaml \  #thick, thin, medium
    my_dataset/visual_test_source/ \
    my_dataset/visual_test/random_<size>_512/ \ #thick, thin, medium
    --ext jpg


    ls my_dataset/visual_test/random_thick_512/
    image1_crop000_mask000.png
    image1_crop000.png
    image2_crop000_mask000.png
    image2_crop000.png
    ...

    # 对 eval_source 图像文件夹执行相同操作：

    python3 bin/gen_mask_dataset.py \
    $(pwd)/configs/data_gen/random_<size>_512.yaml \  #thick, thin, medium
    my_dataset/eval_source/ \
    my_dataset/eval/random_<size>_512/ \ #thick, thin, medium
    --ext jpg



    # 生成定位这些文件夹的配置文件：

    touch my_dataset.yaml
    echo "data_root_dir: $(pwd)/my_dataset/" >> my_dataset.yaml
    echo "out_root_dir: $(pwd)/experiments/" >> my_dataset.yaml
    echo "tb_dir: $(pwd)/tb_logs/" >> my_dataset.yaml
    mv my_dataset.yaml ${PWD}/configs/training/location/


    # 检查数据配置与 my_dataset 文件夹结构的一致性：
    $ cat ${PWD}/configs/training/data/abl-04-256-mh-dist
    ...
    train:
      indir: ${location.data_root_dir}/train
      ...
    val:
      indir: ${location.data_root_dir}/val
      img_suffix: .png
    visual_test:
      indir: ${location.data_root_dir}/visual_test
      img_suffix: .png


    # 运行训练
    python3 bin/train.py -cn lama-fourier location=my_dataset data.batch_size=10

    # 评估：LaMa 训练过程会根据 my_dataset/val/ 上的分数
    # 挑选最佳的几个模型

    # 要在之前未见过的 my_dataset/eval 上评估您的最佳模型之一（例如 epoch=32）
    # 对粗、细和中等掩码执行以下操作：

    # 推理：
    python3 bin/predict.py \
    model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier_/ \
    indir=$(pwd)/my_dataset/eval/random_<size>_512/ \
    outdir=$(pwd)/inference/my_dataset/random_<size>_512 \
    model.checkpoint=epoch32.ckpt

    # 指标计算：
    python3 bin/evaluate_predicts.py \
    $(pwd)/configs/eval2_gpu.yaml \
    $(pwd)/my_dataset/eval/random_<size>_512/ \
    $(pwd)/inference/my_dataset/random_<size>_512 \
    $(pwd)/inference/my_dataset/random_<size>_512_metrics.csv


**或者**在 Docker 中：

    待完成：训练
    待完成：评估

# 提示

### 生成不同类型的掩码
以下命令将执行一个生成随机掩码的脚本。

    bash docker/1_generate_masks_from_raw_images.sh \
        configs/data_gen/random_medium_512.yaml \
        /输入图像目录 \
        /存储图像和掩码的目录 \
        --ext png

测试数据生成命令会以适合[预测](#prediction)的格式存储图像。

下表描述了我们用来生成论文中不同测试集的配置。
请注意，我们*没有固定随机种子*，因此每次结果会略有不同。

|        | Places 512x512         | CelebA 256x256         |
|--------|------------------------|------------------------|
| 窄掩码 | random_thin_512.yaml   | random_thin_256.yaml   |
| 中等掩码 | random_medium_512.yaml | random_medium_256.yaml |
| 宽掩码 | random_thick_512.yaml  | random_thick_256.yaml  |

您可以随意更改配置文件路径（第 1 个参数）为 `configs/data_gen` 中的任何其他配置，
或自行调整配置文件。

### 覆盖配置中的参数
您还可以像这样覆盖配置中的参数：

    python3 bin/train.py -cn <config> data.batch_size=10 run_title=my-title

其中 .yaml 文件扩展名可省略

### 模型选项
论文中模型的配置名称（替换到训练命令中）：

    * big-lama
    * big-lama-regular
    * lama-fourier
    * lama-regular
    * lama_small_train_masks

它们位于 configs/training/ 文件夹中

### 链接
- 所有数据（模型、测试图像等）https://disk.yandex.ru/d/AmdeG-bIjmvSug
- 论文中的测试图像 https://disk.yandex.ru/d/xKQJZeVRk5vLlQ
- 预训练模型 https://disk.yandex.ru/d/EgqaSnLohjuzAg
- 感知损失模型 https://disk.yandex.ru/d/ncVmQlmT_kTemQ
- 我们的训练日志可在此查看 https://disk.yandex.ru/d/9Bt1wNSDS4jDkQ


### 训练时间与资源

待完成

## 致谢

* 分割代码和模型来自 [CSAILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch)。
* LPIPS 指标来自 [richzhang](https://github.com/richzhang/PerceptualSimilarity)
* SSIM 来自 [Po-Hsun-Su](https://github.com/Po-Hsun-Su/pytorch-ssim)
* FID 来自 [mseitzer](https://github.com/mseitzer/pytorch-fid)

## 引用
如果您觉得此代码有帮助，请考虑引用：
```
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```

## Dataset

The overall directory structure should be (taken from [Point-BERT](https://github.com/lulutang0608/Point-BERT)):

```
│Point-BERT/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ShapeNet55-34/
├──.......
```
**ModelNet Dataset:** You can download the processed ModelNet data from [[Google Drive]](https://drive.google.com/drive/folders/1fAx8Jquh5ES92g1zm2WG6_ozgkwgHhUq?usp=sharing)[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/d/4808a242b60c4c1f9bed/)[[BaiDuYun]](https://pan.baidu.com/s/18XL4_HWMlAS_5DUH-T6CjA )(code:4u1e) and save it in `data/ModelNet/modelnet40_normal_resampled/`. (You can download the offical ModelNet from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip), and process it by yourself.) Finally, the directory structure should be:
```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```


**ShapeNet55/34 Dataset:** You can download the processed ShapeNet55/34 dataset at [[BaiduCloud](https://pan.baidu.com/s/16Q-GsEXEHkXRhmcSZTY86A)] (code:le04) or [[Google Drive](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)]. Unzip the file under `ShapeNet55-34/`. The directory structure should be

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-35/
│  ├── train.txt
│  └── test.txt
```

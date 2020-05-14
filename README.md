# CCPD (Chinese City Parking Dataset, ECCV)

## UPdate on 10/03/2019. CCPD Dataset is now updated. We are confident that images in subsets of CCPD is much more challenging than before with over 300k images and refined annotations. 

(If you are benefited from this dataset, please cite our paper.) 
It can be downloaded from and extract by (tar xf CCPD2019.tar.xz):
 - [Google Drive](https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc) 
 
 - [BaiduYun Drive(code: hm0u)](https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw)


#### train\val\test split
The split file is available under 'split/' folder.

Images in CCPD-Base is split to train/val set. Sub-datasets (CCPD-DB, CCPD-Blur, CCPD-FN, CCPD-Rotate, CCPD-Tilt, CCPD-Challenge) in CCPD are exploited for test.

### metric
As each image in CCPD contains only a single license plate (LP). Therefore, we do not consider recall and concerntrate on precision. Detectors are allowed to predict only one bounding box for each image.

- Detection. For each image, the detector outputs only one bounding box. The bounding box is considered to be correct if and only if its IoU with the ground truth bounding box is more than 70% (IoU > 0.7). Also, we compute AP on the test set. 

- Recognition. A LP recognition is correct if and only if all characters in the LP number are correctly recognized.

#### benchmark

If you want to provide more baseline results or have problems about the provided results. Please raise an issue.
##### detection

|             | FPS |   AP  |   DB  |  Blur |   FN  | Rotate |  Tilt | Challenge |
|---|---|---|---|---|---|---|---|---|
| Faster-RCNN |  11 | 84.98 | 66.73 | 81.59 | 76.45 |  94.42 | 88.19 |   89.82   |
|    SSD300   |  25 | 86.99 | 72.90 | 87.06 | 74.84 |  96.53 | 91.86 |   90.06   |
|    SSD512   |  12 | 87.83 | 69.99 | 84.23 | 80.65 |  96.50 | 91.26 |   92.14   |
|  YOLOv3-320 |  52 | 87.23 | 71.34 | 82.19 | 82.44 |  96.69 | 89.17 |   91.46   |

##### recognition 
We provide baseline methods for recognition by appending a LP recognition model Holistic-CNN (HC) (refer to paper 'Holistic recognition of low quality license plates by cnn using track annotated data') to the detector.

|             | FPS |   AP  |   DB  |  Blur |   FN  | Rotate |  Tilt | Challenge |
|---|---|---|---|---|---|---|---|---|
| SSD512+HC |  11 | 43.42 | 34.47 | 25.83 | 45.24 |  52.82 | 52.04 |   44.62   |

The column 'AP' shows the precision on all the test set. The test set contains six parts: DB(ccpd_db/), Blur(ccpd_blur), FN(ccpd_fn), Rotate(ccpd_rotate), Tilt(ccpd_tilt), Challenge(ccpd_challenge).

This repository is designed to provide an open-source dataset for license plate detection and recognition, described in _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_. This dataset is open-source under MIT license. More details about this dataset are avialable at our ECCV 2018 paper (also available in this github) _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_. If you are benefited from this paper, please cite our paper as follows:

```
@inproceedings{xu2018towards,
  title={Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline},
  author={Xu, Zhenbo and Yang, Wei and Meng, Ajin and Lu, Nanxue and Huang, Huan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={255--271},
  year={2018}
}
```



## Specification of the categorise above:

- **rpnet**: The training code for a license plate localization network and an end-to-end network which can detect the license plate bounding box and recognize the corresponding license plate number in a single forward. In addition, demo.py and demo folder are provided for playing demo.

- **paper.pdf**: Our published eccv paper.


## Demo

Demo code and several images are provided under rpnet/ folder, after you obtain "fh02.pth" by downloading or training, run demo as follows, the demo code will modify images in rpnet/demo folder and you can check by opening demo images.

```

  python demo.py -i [ROOT/rpnet/demo/] -m [***/fh02.pth]

```

### The nearly well-trained model for testing and fun (Short of time, trained only for 5 epochs, but enough for testing): 

We encourage the comparison with SOTA detector like FCOS rather than RPnet as the architecture of RPnet is very old fashioned.
- Location module wR2.pth [google_drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-), [baiduyun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw)
- rpnet model fh02.pth [google_drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv), [baiduyun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ).

## Training instructions

Input parameters are well commented in python codes(python2/3 are both ok, the version of pytorch should be >= 0.3). You can increase the batchSize as long as enough GPU memory is available.

#### Enviorment (not so important as long as you can run the code): 

- python: pytorch(0.3.1), numpy(1.14.3), cv2(2.4.9.1). 
- system: Cuda(release 9.1, V9.1.85)

#### For convinence, we provide a trained wR2 model and a trained rpnet model, you can download them from google drive or baiduyun.



First train the localization network (we provide one as before, you can download it from [google drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-) or [baiduyun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw)) defined in wR2.py as follows:

```

  python wR2.py -i [IMG FOLDERS] -b 4

```

After wR2 finetunes, we train the RPnet (we provide one as before, you can download it from [google drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv) or [baiduyun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ)) defined in rpnet.py. Please specify the variable wR2Path (the path of the well-trained wR2 model) in rpnet.py.

```

  python rpnet.py -i [TRAIN IMG FOLDERS] -b 4 -se 0 -f [MODEL SAVE FOLDER] -t [TEST IMG FOLDERS]

```



## Test instructions

After fine-tuning RPnet, you need to uncompress a zip folder and select it as the test directory. The argument after -s is a folder for storing failure cases. 

```

  python rpnetEval.py -m [MODEL PATH, like /**/fh02.pth] -i [TEST DIR] -s [FAILURE SAVE DIR]

```

## Dataset Annotations

Annotations are embedded in file name.

A sample image name is "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg". Each name can be splited into seven fields. Those fields are explained as follows.

- **Area**: Area ratio of license plate area to the entire picture area.

- **Tilt degree**: Horizontal tilt degree and vertical tilt degree.

- **Bounding box coordinates**: The coordinates of the left-up and the right-bottom vertices.

- **Four vertices locations**: The exact (x, y) coordinates of the four vertices of LP in the whole image. These coordinates start from the right-bottom vertex.

- **License plate number**: Each image in CCPD has only one LP. Each LP number is comprised of a Chinese character, a letter, and five letters or numbers. A valid Chinese license plate consists of seven characters: province (1 character), alphabets (1 character), alphabets+digits (5 characters). "0_0_22_27_27_33_16" is the index of each character. These three arrays are defined as follows. The last character of each array is letter O rather than a digit 0. We use O as a sign of "no character" because there is no O in Chinese license plate characters.
```
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
```

- **Brightness**: The brightness of the license plate region.

- **Blurriness**: The Blurriness of the license plate region.



## Acknowledgement

If you have any problems about CCPD, please contact detectrecog@gmail.com.



Please cite the paper _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_, if you benefit from this dataset.

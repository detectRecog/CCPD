# CCPD (Chinese City Parking Dataset, ECCV)

## UPdate on 10/03/2019. CCPD Dataset is now updated. We are confident that images in subsets of CCPD is much more challenging than before with over 300k images and refined annotations. 

(If you are benefited from this dataset, please cite our paper.) 
It can be downloaded from and extract by (tar xf CCPD2019.tar.xz):
 - [Google Drive](https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc) 
 
 - [BaiduYun Drive(code: hm0u)](https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw)


#### Training/Testing/Validation Split

The relevant splits are available under the 'split/' folder once the dataset is downloaded and extracted.

Images in CCPD-Base are split to training/validation set. Sub-datasets (CCPD-DB, CCPD-Blur, CCPD-FN, CCPD-Rotate, CCPD-Tilt, CCPD-Challenge) in CCPD are used for testing.
****
## Update on 16/09/2020. We add a new energy vehicle sub-dataset (CCPD-Green) which has an eight-digit license plate number.

It can be downloaded from: 
 - [Google Drive](https://drive.google.com/file/d/1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR/view?usp=sharing) 
 
 - [BaiduYun Drive(code: ol3j)](https://pan.baidu.com/s/1JSpc9BZXFlPkXxRK4qUCyw)
  
### Metrics
Each image in CCPD contains only a single license plate (LP). Therefore, we do not consider recall and concentrate on precision. Detectors are allowed to predict only one bounding box for each image.

- Detection: For each image, the detector outputs only one bounding box. The bounding box is considered to be correct if and only if its IoU with the ground truth bounding box is more than 70% (IoU > 0.7). Also, we compute AP on the test set. 

- Recognition: A LP recognition is correct if and only if all characters in the LP number are correctly recognized.

#### Benchmark

If you want to provide more baseline results or have problems about the provided results. Please raise an issue.
##### Detection

|             | FPS |   AP  |   DB  |  Blur |   FN  | Rotate |  Tilt | Challenge |
|---|---|---|---|---|---|---|---|---|
| Faster-RCNN |  11 | 84.98 | 66.73 | 81.59 | 76.45 |  94.42 | 88.19 |   89.82   |
|    SSD300   |  25 | 86.99 | 72.90 | 87.06 | 74.84 |  96.53 | 91.86 |   90.06   |
|    SSD512   |  12 | 87.83 | 69.99 | 84.23 | 80.65 |  96.50 | 91.26 |   92.14   |
|  YOLOv3-320 |  52 | 87.23 | 71.34 | 82.19 | 82.44 |  96.69 | 89.17 |   91.46   |

##### Recognition 
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

## Repository Structure
```
├── README.md
├── Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf
├── .gitignore
├── requirements.txt
├── src
│   ├── data
│   │   ├── datasets.py --> Dataset classes
│   │   ├── setup_utils.py --> Setup environment variables (etc..) 
│   ├── modules
│   │   ├── detection.py --> localization network
│   │   ├── recognition.py --> end-to-end recognition network
├── LICENSE


## Setup
### Python environment
```bash
  conda create -n ccpd python=3.11 -y && activate ccpd && pip install -r requirements.txt
```

### Dataset
1. Download the full dataset and extract it
2. Point the environment `DATA_DIR` variable set in `src/data/setup_utils.py` to the dataset path (i.e., the directory containing the directory `CCPD2019`)

### Pretraining
```bash
python src/pretrain.py --batch_size 64 --max_epochs 100 --use_gpu --devices 0,1,2,3
```

### Training
```bash
python src/train.py --batch_size 64 --max_epochs 100 --use_gpu --devices 0,1,2,3 --pretrained_model_path [path to pretrained model .ckpt file]
```

### Testing
Testing is done at the end of training. 

### The nearly well-trained model for testing and fun (Short of time, trained only for 5 epochs, but enough for testing): 

We encourage the comparison with SOTA detector like FCOS rather than RPnet as the architecture of RPnet is very old fashioned.
- Location module wR2.pth [google_drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-), [baiduyun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw)
- rpnet model fh02.pth [google_drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv), [baiduyun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ).

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

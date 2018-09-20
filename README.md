# CCPD



Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline



# CCPD: Chinese City Parking Dataset

<span style="color:red">**Attention: there are some errors in dataloader and this is why training goes wrong. These two days I will correct these errors and uploaded a refined dataset && a test code && a well trained rpnet model.**</span>

This repository is designed to provide an open-source dataset for license plate detection and recognition, described in _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_. 

## The google drive link for directly downloading the whole dataset: [google drive 12GB](https://drive.google.com/open?id=1fFqCXjhk7vE9yLklpJurEwP9vdLZmrJd). 

## The baiduyun link for directly downloading the whole dataset: [baiduyun 12GB](https://pan.baidu.com/s/1FH6pFOFF2MwyWiqn6vCzGA).

## The nearly well-trained model for testing and fun (Short of time, trained only for 5 epochs, but enough for testing): [Location module wR2.pth google_drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-), [Location module wR2.pth baiduyun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw), [rpnet model google_drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv), and [rpnet model baiduyun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ).

This dataset is open-source under MIT license. Files under this git repo are sample images. More details about this dataset are avialable at ECCV 2018 paper _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_. If you are benefited from this paper, please cite our paper as follows:

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



- **sample**: gives 6 example pictures for each sub-dataset(blur/challenge/db/fn/np/rotate/tilt).



- **rpnet**: The training code for a license plate localization network and an end-to-end network which can detect the license plate bounding box and recognize the corresponding license plate number in a single forward.


## Annotations

Annotations are embedded in file name.

A sample image name is "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg". Each name can be splited into seven fields. Those fields are explained as follows.

- **Area**: Area ratio of license plate area to the entire picture area.

- **Tilt degree**: Horizontal tilt degree and vertical tilt degree.

- **Bounding box coordinates**: The coordinates of the left-up and the right-bottom vertices.

- **Four vertices locations**: The exact (x, y) coordinates of the four vertices of LP in the whole image. These coordinates start from the right-bottom vertex.

- **License plate number**: Each image in CCPD has only one LP. Each LP number is comprised of a Chinese character, a letter, and five letters or numbers.

- **Brightness**: The brightness of the license plate region.

- **Blurriness**: The Blurriness of the license plate region.



## Training instructions

Input parameters are well commented in python codes(python2/3 are both ok, the version of pytorch should be >= 0.3). You can increase the batchSize as long as enough GPU memory is available.



#### For convinence, we provide a well-trained wR2 model named "wR2.pth221" in the rpnet/ folder for easy training RPnet.



First train the localization network (we provide one as before, you can download it from [google drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-) or [baiduyun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw)) defined in wR2.py as follows:

```

  python wR2.py -i [IMG FOLDERS] -b 4

```

After wR2 finetunes, we train the RPnet (we provide one as before, you can download it from [google drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv) or [baiduyun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ)) defined in rpnet.py. Please specify the variable wR2Path (the path of the well-trained wR2 model) in rpnet.py.

```

  python rpnet.py -i [TRAIN IMG FOLDERS] -b 4 -se 0 -f [MODEL SAVE FOLDER] -t [TEST IMG FOLDERS]

```



## Test demo instructions

After fine-tuning RPnet, you need to uncompress a zip folder and select it as the test directory. The argument after -s is a folder for storing failure cases. File rpnetEval.py is for testing thousands of images in the test directory and print precision. File rpnetTestSeveralImages.py is for evaluating several images in a folder and plot the results on images.

```

  python rpnetEval.py -m [MODEL PATH, like /**/fh02.pth] -i [TEST DIR] -s [FAILURE SAVE DIR]

```



## Acknowledgement

If you have any problems about CCPD, please contact detectrecog@gmail.com.



Please cite the paper _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_, if you benefit from this dataset.

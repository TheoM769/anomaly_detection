### You can load your datasets here

For this project, datasets need to respect MVTec format : 

```
└─ Dataset/object_catagory
   ├── train/
   │   ├── good/ # directory with list of good images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   ├── test/
   │   ├── good/ # directory with list of good images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   │   ├── defective1/ # directory with list of defective images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   │   ├── defective2/ # directory with list of defective images
   │   │   ├── img1.png
   │   |   ├── img2.png
   │   |   └── ...
   └── ground_truth/ # directory with semantic segmentation masks
       ├── defective1/ # directory with list of defective images for detection and segmentation task
       │   ├── img1_mask.png
       |   ├── img2_mask.png
       |   └── ...
       ├── defective2/ # directory with list of defective images for detection and segmentation task
       │   ├── img1_mask.png
       |   ├── img2_mask.png
       |   └── ...
```

As an example dataset, see : https://www.mvtec.com/company/research/datasets/mvtec-ad.
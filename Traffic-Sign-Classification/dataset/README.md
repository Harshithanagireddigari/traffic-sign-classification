# Dataset folder

Download the **German Traffic Sign Recognition Benchmark (GTSRB)** and extract
it into this folder so the final layout looks like:

```
dataset/
└── GTSRB/
    ├── Final_Training/
    │   └── Images/
    │       ├── 00000/   (PPM images + GT-00000.csv)
    │       ├── 00001/
    │       └── ...      (43 class folders, 00000 - 00042)
    └── Final_Test/
        └── Images/
            ├── GT-Final_Test.csv  (or Test.csv depending on the archive)
            └── *.ppm test images
```

## Where to download GTSRB

- Official benchmark page: https://benchmark.ini.rub.de/gtsrb_dataset.html
  (download `GTSRB_Final_Training_Images.zip` and `GTSRB_Final_Test_Images.zip`)
- Kaggle mirror (easier, needs a free account):
  https://www.kaggle.com/datasets/meowmeowmeowmeowmeowmeow/gtsrb-german-traffic-sign

> The dataset contains ~39,000 training images and ~12,800 test images in
> PPM format across 43 sign classes. OpenCV (`cv2.imread`) reads PPM files
> directly, so no conversion is needed.

`train.py` expects exactly the structure above and will print a clear error
message if the dataset is missing.

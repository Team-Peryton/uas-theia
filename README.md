# UAS Theia

Surrey IMechE UAS 2022 Image Recognition program

Interfaces with Nyx like:

```python

results : List[ImageRecognitionResult] = theia.image_recognition(image : np.ndarray, gps_image_taken: Tuple(float, float))

# where

@dataclass
class ImageRecognitionResult:
    image_name: str = "" # file name
    centre: Tuple[int,int] = (0, 0)
    position: Tuple[float,float] = (0.0, 0.0) # lat, lon
```

## Dataset setup

dataset download: https://www.sensefly.com/education/datasets/?dataset=1502

should be extracted into the dataset/raw_dataset directory then the datset/generate.ipynb notebook should be run

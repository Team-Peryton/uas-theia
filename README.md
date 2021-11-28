# UAS Theia

Surrey IMechE UAS 2022 Image Recognition program

Interfaces with Nyx like:

```python

results : List[ImageRecognitionResult] = theia.image_recognition(image : np.ndarray)

# where

@dataclass
class ImageRecognitionResult:
    image_name: str = "" # file name
    charachter: str = ""
    colour: Tuple[int,int,int] = (0, 0, 0) # [R, G, B]
    centre: Tuple[int,int] = (0, 0)
    position: Tuple[float,float] = (0.0, 0.0) # lat, lon
    cropped: np.ndarray = np.array([]) # the cropped image
    no_nested_square: bool = False

```

dataset download: https://www.sensefly.com/education/datasets/?dataset=1502

# Augmented Reality

## Implementation details

- Conducted camera calibration of my phone using Zhang's algorithm, accurately determining the intrinsic and extrinsic parameters.
- Leveraged the obtained calibration results to place artificial objects onto images.

Check the report for more detailed explanation.

## Setup

- Install the required libraries
```
pip install -r requirements.txt
```

## Running the code

- Place at least 3 images of a checkerboard taken from different angles (example provided in images directory) in the same path as `camera_calibration.py`.
-  Then run `camera_calibration.py`
```
python camera_calibration.py
```
- View the results generated in the `results` folder.
- Example images
![alt text](https://github.com/shashank-g12/Computer-Vision/blob/main/Augmented%20Reality/results/image_2.jpg)
-
![alt text](https://github.com/shashank-g12/Computer-Vision/blob/main/Augmented%20Reality/results/image_3.jpg)


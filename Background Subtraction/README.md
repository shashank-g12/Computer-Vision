## Adaptive background mixture models for real-time tracking using exponentially decaying weights.

### Data format
```
├── Candela_m1.10
│   └── input
│       |── Candela_m1.10_000000.png
|       |── Candela_m1.10_000001.png
|       ...
|       └── Candela_m1.10_000349.png
```
### Parameters used are as follows-
  - `K`: No of gaussian mixture models for a particular pixel (default = 5)
  - `lr`: Learning rate (default = 0.003)
  - `T`: Threshold used to determine the background model for a pixel (default = 0.8)
  - `C`: Minimum no of connected component size (pixels are displayed only if it is a part of a connected component of size greater than value). Used to eliminate noise. Used Two-pass connected component algorithm (8-connectivity) as mentioned in the paper. (default = 50)

### Running the code 
- First, install the opencv module
- Compile the code with the command
```
g++ AdaptiveGMM.cpp -o exec `pkg-config --cflags --libs opencv4`
```
- run the executable
```
./exec
```

### Failure methods:
- Detection of shadows of moving objects. This is one of the disadvantages of this algorithm which is solved in the 'non parametric model for background subtraction' paper.
- Detection of plant in the Candela_m1.10 dataset.
- Still image of moving objects for certain frames at the start. (for Eg in the HighwayI dataset, some cars remain still for a few frames at the start of the video.)


### Analysis of comparison with Non parametric model for background subtraction with constant weights(1).
- Fewer shadows pixels detected in (1) due to the scaling of R,G,B values as mentioned in the paper by which color information is separated from the lightness information which is used to suppress shadows.
- The moving objects adapt quicker in (1) than in our method due to the usage of short term model and sample values itself as mean for intensity distribution.
- Less false positive detection in (1), For Eg, In the case of Candela_m1.10 dataset even the plant was being deteced in our method but not in (1).


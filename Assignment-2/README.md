# Image Stiching (Panaroma)

## Implementation details

- Implemented Harris corner detection as illustrated in the paper 'A COMBINED CORNER AND EDGE DETECTOR' by Chris Harris and Mike Stephens.
After finding the corner strength of each pixel using det(H)-k*Trace(H)^2 (k=0.1), I have performed non maxima suppression of size 7x7 to get the local maxima. Then I am picking top N corner strengths as features for the image and using the pixel neighbourhood values as feature descriptor which is of size 64.


- After finding the feature and feature descriptors for two adjacent frames, matching is performed using proximity(nearest neighbour) using sum of squared differences approach with a threshold set to 10^5.


- An affine matrix was calculated using the matched keypoints with the help of ransac algorithm (used opencv function estimateAffine2d() for this). Then the adjacent frames were stiched using opencv warpAffine function and a combine function writen by myself.I have Implemented pairwise Stiching where it stiches image 1 with image 2 and then the resultant with image 3 and so on.

## Dataset setup

├── Dataset
│    |── 1.png
|    |── 2.png
|    ...
|    └── 10.png

## Running the code

- Compile the code using the command
```
g++ harris.cpp -o exec `pkg-config --cflags --libs opencv4`
```
- Run the executable
```
./exec
```
- Stiched image is `stiched.png`


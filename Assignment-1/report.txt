Name: Shashank G
Entry No: 2022AIB2684


Method used : Y=0 Adaptive background mixture models for real-time tracking using exponentially decaying weights.
Parameters used are as follows-
K = 5           -   No of gaussian mixture models for a particular pixel.
lr = 0.003      -   Learning rate.
T = 0.8         -   Threshold used to determine the background model for a pixel.
C = 50          -   Minimum no of connected component size (pixels are displayed only if it is a part of a connected component of size greater than 50). Used to eliminate noise. Used Two-pass connected component algorithm (8-connectivity) as mentioned in the paper.


Failure methods:
i) Detection of shadows of moving objects. This is one of the disadvantages of this algorithm which is solved in the 'non parametric model for background subtraction' paper.
ii) Detection of plant in the Candela_m1.10 dataset.
iii) Still image of moving objects for certain frames at the start. (for Eg in the HighwayI dataset, some cars remain still for a few frames at the start of the video.)


Analysis of comparison with 2022AIB2687 Y=3, Non parametric model for background subtraction with constant weights.
i) Fewer shadows pixels detected in Y=3 due to the scaling of R,G,B values as mentioned in the paper by which color information is separated from the lightness information which is used to suppress shadows.
ii) The moving objects adapt quicker in Y=3 than in Y=0 due to the usage of short term model and sample values itself as mean for intensity distribution.
iii) Less false positive detection in Y=3, For Eg, In the case of Candela_m1.10 dataset even the plant was being deteced in Y=0 but not in Y=3.


Link of one drive for output videos: https://csciitd-my.sharepoint.com/:f:/g/personal/aib222684_iitd_ac_in/EqJpLlW4LaxHhXmpbIW2s1sB1Up7T-BTh4En4_J50UBw8Q
(Will have to download the video to view it as microsoft one drive is not able to load the video).

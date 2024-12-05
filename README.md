# Image Processing Algorithms

  This C++ code implements two image processing algorithms using OpenCV: block matching with motion compensation and weighted averaging. The code processes a sequence of raw images and produces deghosted images using block matching and an averaged image using weighted averaging.

  # Block Matching Algorithm

  The block_matching function reads a sequence of raw images, performs block matching between consecutive frames, estimates motion vectors, and compensates for motion to create deghosted images. The processed images are saved as deghosted_image.raw.

# Weighted Averaging Algorithm

  The weighted_averaging function reads a set of raw images, converts them to floating-point format, performs weighted averaging, and saves the resulting averaged image as averaged_image.raw. 

# Usage

1. Adjust the width and height variables in the code according to the dimensions of the input images.
2. Ensure that raw images are named as "img1.raw," "img2.raw," ..., and are placed in the "im6" directory.
3. Compile and run the code.

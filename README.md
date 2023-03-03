# PyTorch accelerated Particle Image Velocimetry
This program implements the basic algorithms of the PIV method, such as an iterative cross-correlation method based on FFT with an integer and continuous displacement __(DWS, CWS)__ of the interrogation windows, filtering and interpolation of the pair loss effect, and so on. At this stage, the __graphical interface__ is available, the ability to select PIV hyperparameters. The key feature of the project is the use of graphics accelerators due to the __torch__ library. PIV algorithm is completely vectorized, what results in a very high performance using the GPU, but still has a room for improvement.

__Parameters of the program:__
1. Interrogation window size
2. Window overlap size
3. Coordinate scale
4. Time between frames
5. Selection of the device (CPU or any of your GPUs)
6. Iteration method: DWS or CWS
7. The number of iterations of the algorithm
8. Interrogation window scale during iterations
9. Supported image formats (bmp, jpg, tiff, ect.)
10. Options for saving program results

__Installation:__    
It is easier to use conda environment.
1. Install nvidia CUDA Toolkit https://developer.nvidia.com/cuda-toolkit to ensure latest nvidia driver usage
2. Install pytorch with GPU support https://pytorch.org/ . Matching your CUDA version is not critical since PyTorch installs it's own cudatoolkit.  
3. In your environment or command line <code>pip install -r /path/to/requirements.txt </code>
If you facing some troubles try to <code>pip install "module"</code> manually for each line

Tested on Windows

__Performance__:  
This method allows processing 4 thousand pairs of images with a size of 4 MP each with a search window of 64, overlap of 50%, two iterations with re-arranging (an increase in the number of vectors by 4 times) in less than 10 minutes. The first iteration of the algorithm (~[4000, 64, 64] subimage tensor) takes ~15 ms on a Geforce GTX 1660 Ti GPU.

__Futhure plans__:
Creating a version of the program for processing data during their recording. The ability to process stereo PIV data.

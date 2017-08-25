# MasterThesis

This is the accompanying code for Maastricht University Master Thesis DKE 17-15

To execute this code, install python 2.7 including 
* numpy
* opencv
* scikit-learn
* dlib

Anaconda/Miniconda are recommended

To read video files, opencv sources need to be downloaded and compiled by hand with ffmpeg support. There are currently no opencv distributions with built-in support due to license-imposed restrictions.

There are three main executables to replicate the results in the master thesis, located on top level.
All executables build on top of each other, *modify* and run in the following order:

* ExtractLandmarks.py
* ExtractFeatures.py
* BagOfWordsClassification.py 

The modifications needed are:
* Download the model
   * The facial landmark detection model (shape_predictor_68_face_landmarks.dat) needs to be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2. Make a 'models' folder in 'operators' and put the model there.
* Setting the paths. 
   * There are parameters for the paths in the main functions, but they are hard coded in the `if __name__=='__main__'` call.
   * For BagOfWordsClassification.py, the paths are completely hard coded and located at the top of the file
* Setting the logging mode
   * To prevent expensive computations, like iterating through 700 video files, landmarks and features are cached. 
   * The `logging_mode` parameter controls caching: 
      * `LIVE` : Processing happens every time and nothing is written
      * `RECORD` : Processing happens every time and log files are written
      * `REPLAY` : Log files are read, no processing happens
 
ExtractLandmarks and ExtractFeatures both have a secondary purpose build-in.

When `activate_view` is set to `True` in ExtractLandmarks, windows showing the landmark detection results and feature extraction are shown. This feature is 'experimental' meaning that while it is theoretically possible to press escape when focusing the landmark window to exit, it might not work every time. Pressing the right arrow instead skips to the next video.

When `output_folder` is not set to `None` in ExtractFeatures, the clustering parameters become active. Depending on the parameters, agglomeraive clustering or DBSCAN using either euclidean or pseudo cosine distance is performed using the features. The output is written to the output folder. The choice of algorithm depends on the parameters. If `agglomerative_n` is `None`, DBSCAN is performed. If all clustering parameters are `None`, the script will fail with an assertion.

BagOfWordsClassification should be self explanatory. Modify the paths and run.

# Classification of response to publicity through emotion detection and gaze tracking

The goal of this project is to give an efficient solution to classificate the impact of printed publicity, through the detection of the emotion expressed by the viewers, and also through the analysis of the point of the publicity that's being observed.

The folder "models" contains the file "emotion_model.hdf5", that represents the model with corresponding weights and architecture used for emotions detection.

On the other hand, there is also the folder "gaze_tracking", where the declaration of the functions is made, for the calculation of the gaze position. On this file, the functions that decide where are the eyes directed are contained as well. The result of where the eyes are pointing depends on a fixed threhshold, also defined on the file "gaze_tracking.py".

The file "emotions.py" and "emotions_RS.py" contain the implementation of this function, and they execute the detection taking as input the webcam of a computer, or an Intel camera compatible with the library pyrealsense2. The mounted system was tested using cameras D435i and SR300, both with correct results.

# AISclassification

This source code is related to the work in [[1]](https://www.mdpi.com/1424-8220/22/16/6063#cite). Thus, if you are using this code please cite [[1]](https://www.mdpi.com/1424-8220/22/16/6063#cite).
The arxiv file is available at [link](https://arxiv.org/abs/2207.05514v1).

A geometric-driven semi-supervised approach for fishing activity detection from AIS data. 
It employes a cluster analysis to label the vessel’s movement pattern by indicating fishing activity, exploring the geometry of the vessel route included in the messages.
Next, the proposed recurrent neural networks is applied to detect fishing activities on AIS data streams with roughly 87% of the overall F-score on the whole trajectories of 50 different unseen fishing vessels.
This source code also includes a broad benchmark study assessing the performance of different Recurrent Neural Network (RNN) architectures.

## Files Description

observation-based.py: it executes the clustering based on the number of messages.

time-based.py: it executes the clustering based on the time between messages.

distance-based.py: it executes the clustering based on the distance between messages.

network/run.py: it executes the proposed RNN.

### Read data
1. Read data from NOAA
2. Clean data
   1. remove messages with invalid cog
   2. remove msg that vessel are not moving
   3. remove trajectories with less than 100 messages
3. Get trajectories from the Juan strait

Label data using clustering algorithms
1. Get features using MA in the SOG and COG
2. Clustering using kmeans

Apply time series classification

## Requirements

colorama==0.4.4\
cycler==0.11.0\
Cython==0.29.28\
fonttools==4.31.2\
haversine==2.5.1\
joblib==1.1.0\
kiwisolver==1.4.2\
matplotlib==3.5.1\
numpy==1.22.3\
packaging==21.3\
pandas==1.4.1\
patsy==0.5.2\
Pillow==9.0.1\
pyparsing==3.0.7\
python-dateutil==2.8.2\
pytz==2022.1\
scikit-learn==1.0.2\
scikit-posthocs==0.6.7\
scipy==1.8.0\
seaborn==0.11.2\
six==1.16.0\
statsmodels==0.13.2\
threadpoolctl==3.1.0\
torch==1.10.1+cu113\
torchaudio==0.10.1+cu113\
torchvision==0.11.2+cu113\
tqdm==4.64.0\
typing_extensions==4.2.0

## Reference

[1] Ferreira, M.D.; Spadon, G.; Soares, A.; Matwin, S. A Semi-Supervised Methodology for Fishing Activity Detection Using the Geometry behind the Trajectory of Multiple Vessels. Sensors 2022, 22, 6063. https://doi.org/10.3390/s22166063

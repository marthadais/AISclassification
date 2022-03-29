# AISclassification

Read data
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

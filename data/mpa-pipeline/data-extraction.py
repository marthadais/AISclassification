#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
import multiprocessing


# In[11]:


from aisdb import DBQuery
from datetime import datetime
from aisdb.database import sqlfcn_callbacks
from tqdm.contrib.concurrent import thread_map, process_map


# In[3]:


warnings.filterwarnings("ignore")


# In[4]:


random_seed = 845369
random.seed(random_seed)
np.random.seed(random_seed)


# In[5]:


if not os.path.exists("pkl/"):
    os.makedirs("pkl/")


# In[6]:


mpa_locations = pd.read_csv("./mpa-locations.csv", index_col=0, encoding="latin-1")[["M2", "X", "Y"]]


# In[7]:


temporal_range = [
    (2015, datetime(year=2015, month=1, day=1), datetime(year=2015, month=12, day=31)),
    (2016, datetime(year=2016, month=1, day=1), datetime(year=2016, month=12, day=31)),
    (2017, datetime(year=2017, month=1, day=1), datetime(year=2017, month=12, day=31)),
]


# In[8]:


def boundingbox_distance(y, x, distance):
    y_func = (distance / 6371000.0) * (180.0 / np.pi)
    x_func = (distance / 6371000.0) * (180.0 / np.pi) / np.cos(y * np.pi / 180.0)
    # Calculates the involving bounding box given a centroid and the square-side size
    return {"ymin": y - y_func, "xmin": x - x_func, "ymax": y + y_func, "xmax": x + x_func}


# In[9]:


def extract_data(data_input):
    rid, mpa = data_input
    for yyyy, t0, t1 in temporal_range:
        vessel_list = DBQuery(
            start=t0, end=t1,  # Search window
            callback=sqlfcn_callbacks.in_bbox_time,
            **boundingbox_distance(  # Search radius
                x=np.float(mpa.X), y=np.float(mpa.Y),
                distance=(np.sqrt(np.float(mpa.M2)) + 20000.0),
            )
        )
        vessel_list.check_idx()  # Check if tables exist and indexes are built
        vessel_list = list(vessel_list.gen_qry())  # Generate and run the query on AISDB
        pkl.dump(vessel_list, open("pkl/%d-%d.pkl" % (rid, yyyy), "wb"), pkl.HIGHEST_PROTOCOL)


# In[ ]:


process_map(extract_data, mpa_locations.iterrows(), max_workers=multiprocessing.cpu_count())

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

df = pd.read_csv('/home/yamaguchi/vmlserver06/Experiment/ATA/output/3sec/eval_P1_1_T3@3.csv')
print(df.head())

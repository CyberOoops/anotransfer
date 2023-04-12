import os
import sys
import pandas as pd
import numpy as np
file_list = os.listdir('./AIOPS')
for file in file_list:
    df = pd.read_csv(os.path.join('./AIOPS',file))
    if np.min(np.unique(np.diff(df['timestamp']))) == 60:
        f1 = os.path.join('./AIOPS',file)
        f2 = os.path.join('./AIOPS1',file)
        os.system('cp {} {}'.format(f1,f2))
    if np.min(np.unique(np.diff(df['timestamp']))) == 300:
        f1 = os.path.join('./AIOPS',file)
        f2 = os.path.join('./AIOPS5',file)
        os.system('cp {} {}'.format(f1,f2))

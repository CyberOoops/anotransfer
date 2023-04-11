import os
import sys
import pandas as pd
file_list = os.listdir('./AIOPS')
for file in file_list:
    df = pd.read_csv(os.path.join('./AIOPS',file))
    
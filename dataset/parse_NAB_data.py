import json
import os
import pandas as pd
import time
import sys
import numpy as np
file_list = os.listdir(sys.argv[1])
with open('./combined_windows.json', encoding='utf-8') as a:
    label_json = json.load(a)
path  = './NAB'
cnt = -1
for file in file_list:
    cnt +=1
    df = pd.read_csv(os.path.join(sys.argv[1],file))
    df['timestamp'] = df['timestamp'].apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    check = np.unique(np.diff(np.asarray([df['timestamp']])))
    print(check)
    df['label'] = 0
    label = label_json[os.path.join(sys.argv[1],file)[15:]]
    interval = df.loc[1,'timestamp'] - df.loc[0,'timestamp']
    now_anomaly_new = []
    for item in label:
        start = int(((time.mktime(time.strptime(item[0][:-7],"%Y-%m-%d %H:%M:%S")))- df.loc[0,'timestamp'])/interval)
        end = int(((time.mktime(time.strptime(item[1][:-7],"%Y-%m-%d %H:%M:%S")))- df.loc[0,'timestamp'])/interval)
        print(start,end)
        df.loc[start:end+1,'label'] = 1
    #print(df)

    df.to_csv('./NAB/{}'.format(file),index = False,mode='w')



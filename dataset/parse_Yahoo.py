import os
import sys
import pandas as pd
file_list = os.listdir(sys.argv[1])
# for f in file_list:
#     if(f[:4]=='real'):
#         df = pd.read_csv(os.path.join(sys.argv[1],f))
#         df['timestamp'] = df['timestamp']*3600+1416722400
#         df.to_csv(os.path.join(sys.argv[1],f),index=False)
l = len(file_list)
for i in range(int(0.3*l)):
    f1 = os.path.join(sys.argv[1],file_list[i])
    f2 = os.path.join(sys.argv[1]+'_cluster',file_list[i])
    os.system('cp {} {}'.format(f1,f2))
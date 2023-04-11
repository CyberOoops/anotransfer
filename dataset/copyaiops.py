import os
file_list = os.listdir('./AIOPS2018')
print(file_list)
os.system('rm -rf ./cluster/*')
os.system('rm -rf ./test/*')
for i in range(len(file_list)):
    if i < int(len(file_list)*0.4):
        src = os.path.join('./AIOPS2018',file_list[i])
        dst = './cluster/'
        os.system('cp {} {}'.format(src,dst))
    else:
        src = os.path.join('./AIOPS2018',file_list[i])
        dst = './test/'
        os.system('cp {} {}'.format(src,dst))
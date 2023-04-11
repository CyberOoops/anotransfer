import os
file = '../AnoTransfer-data/real-world'
file_list = os.listdir(file)
print(file_list)
os.system('rm -rf ./cluster/*')
os.system('rm -rf ./test/*')
for i in range(len(file_list)):
    if i < int(len(file_list)*0.4):
        src = os.path.join(file,file_list[i])
        dst = './cluster/'
        os.system('cp {} {}'.format(src,dst))
    """ else:
        src = os.path.join(file,file_list[i])
        dst = './test/'
        os.system('cp {} {}'.format(src,dst)) """
    src = os.path.join(file,file_list[i])
    dst = './test/'
    os.system('cp {} {}'.format(src,dst))
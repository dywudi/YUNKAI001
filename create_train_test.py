from glob import glob
import os

kinds = os.listdir('cleaned')

for ix, kind in enumerate(kinds):
    files = glob(f'cleaned/{kind}/*.mp4')
    for index, fi in enumerate(files):
        if index == 0:
            with open('traintestsplit/test.txt','a+') as f:
                f.writelines(f'{fi} {ix}\n')
        else:
            with open('traintestsplit/train.txt','a+') as f:
                f.writelines(f'{fi} {ix}\n')
 

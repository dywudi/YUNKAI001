from glob import glob
import json
from tqdm import tqdm
import os

csvs = glob(f'csv/Json/*/*.json')
alist = ['A01','A02','A03','A04','A05','A06','A07','A08','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26','A30','A31']
csvdict = {}
for csv in tqdm(csvs):
    catid = csv.split('/')[-2].split('_')[0]
    catid = alist.index(catid)

    if catid not in csvdict:
        templist = []
        templist.append(csv)
        csvdict[catid] = templist
    else:
        templist = csvdict[catid]
        templist.append(csv)
        csvdict[catid] = templist

for k in tqdm(csvdict.keys()):
    templist = csvdict[catid]
    trainsplit = int(len(templist)*0.8)
    for i in range(len(templist)):
        if i < trainsplit:
            kineket = []
            finalkineket = []
            #print(templist[i])
            currentcsv = templist[i]
            with open(currentcsv,'r') as load_f:
                load_dict = json.load(load_f)
                m = load_dict['file']
                #print(m[0]['frames'])
                for frame in m[0]['frames']:
                    persons = frame['persons']
                    for person in persons:
                        if person['index'] == '0':
                            kineket.append(person['keypoints'])
                            #print(person['keypoints'])
            skip = len(kineket)//32
            
            with open('X_train.txt','a+') as f:
                with open('Y_train.txt','a+') as g:
                    for i in range(0,32):
                        line = kineket[i*32]
                        outputline = ''
                        for item in line:
                            a = item.split(',')
                            outputline = outputline+a[0]+','+a[1]+','

                        f.writelines(f'{outputline}\n')
                        g.writelines(f'{k}\n')
        else:
            kineket = []
            finalkineket = []
            #print(templist[i])
            currentcsv = templist[i]
            with open(currentcsv,'r') as load_f:
                load_dict = json.load(load_f)
                m = load_dict['file']
                #print(m[0]['frames'])
                for frame in m[0]['frames']:
                    persons = frame['persons']
                    for person in persons:
                        if person['index'] == '0':
                            kineket.append(person['keypoints'])
                            #print(person['keypoints'])
            skip = len(kineket)//32
            
            with open('X_test.txt','a+') as f:
                with open('Y_test.txt','a+') as g:
                    for i in range(0,32):
                        line = kineket[i*32]
                        outputline = ''
                        for item in line:
                            a = item.split(',')
                            outputline = outputline+a[0]+','+a[1]+','

                        f.writelines(f'{outputline}\n')
                        g.writelines(f'{k}\n')
    
    
            # for mm in m:
            #     print(mm)
            #     print('\n\n\n')


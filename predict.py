import json
import os
import time

from src.lstm import ActionClassificationLSTM
from src.video_analyzer import analyse_json


# Load pretrained LSTM model from checkpoint file
lstm_classifier = ActionClassificationLSTM.load_from_checkpoint("epoch=399-step=2000.ckpt")
lstm_classifier.eval()

def getpoints(filename):
    kineket = []
    with open(filename,'r') as load_f:
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
    outputline = ''
    for i in range(0,32):
        line = kineket[i*32]
        
        for item in line:
            a = item.split(',')
            outputline = outputline+a[0]+','+a[1]+','
    jsoncontent = [float(i) for i in jsoncontent][0:15]
    return jsoncontent

def analyze(filename,lstm_classifier=lstm_classifier):
    try:
      filename = getpoints(filename)
      result = analyse_json(lstm_classifier, filename)
      return result
    except:
      return False


if __name__ == '__main__':
   #jsonfilename = ''
   #videofilename = ''
   #videoresult = videoanalyze(videofilename)
   jsoncontent = '1828.244507,368.413940,1836.615073,434.783863,1814.393332,419.315532,1794.315166,487.953751,0.000000,0.000000,1874.320685,440.971196,1869.312500,530.215454,1819.341553,497.872681,1810.559570,586.222961,1795.818237,577.468567,1769.331665,698.004028,1737.030273,798.142578,1825.493774,595.078003,1781.229492,703.919373,1742.881714,812.779968'.split(',')
   print(jsoncontent)
   jsoncontent = [float(i) for i in jsoncontent][15:]
   jsonresult = analyze(lstm_classifier,jsoncontent)
   print(jsonresult)
   # if videoresult != jsonresult:
   #    print(jsonresult)
   # else:
   #    print(videoresult)

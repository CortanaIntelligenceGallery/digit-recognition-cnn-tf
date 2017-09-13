import numpy as np
import sys
import urllib
import json
import pandas as pd
from PIL import Image
import array

def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32" )
    return data

def conv_to_single_dim(arr, normalized=True):
    newarr = np.zeros([arr.shape[0], arr.shape[1]])

    newarr = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / 3

    newarr = np.reshape(newarr, [784])
    for i in range(len(newarr)):
        if (normalized):
            newarr[i] = 1 - (newarr[i] / 255)
            newarr[i] = newarr[i] if newarr[i] > 0.5 else 0
        else:
            newarr[i] = (newarr[i] / 255)

    return newarr


def run(nparr):
    data = {"nparr":7}
    str3=np.array2string(nparr, separator=', ').replace('\n','').replace(' ','').replace('0.,','0.0,').replace('1.,','1.0,').replace('0.]','0.0]')
    str4='['+str3+']'
    body = str.encode((json.dumps(data)).replace('7',str4))
    url = "http://52.170.83.20:80/api/v1/service/mnistws/score"
    headers={'Content-Type':'application/json'}
    try:
       req = urllib.request.Request(url, body, headers)
       result = urllib.request.urlopen(req)
       x = result.read().decode("utf-8") 
       y = json.loads(x)
       return y
    except urllib.error.HTTPError as error:
       print("The request failed with status code: " + str(error.code))
       print(error.info())

def predict_img(fname):
    arr = load_image(fname)
    arr = conv_to_single_dim(arr, normalized=False)
    print(fname,": ")
    print(run(arr))

def main():
    global retscores
    print("main")
    
    predict_img("images/2.png")


print("calling main")
main()

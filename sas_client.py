# Copyright <2019> Edward Verenich <verenie@clarkson.edu>
# MIT license <https://opensource.org/licenses/MIT>
# This script iteratively tests models within a 
# specified models folder. 
import os
import requests


# helper method to load images
# load models helper
def load_files(folder_path):
    X = os.listdir(folder_path)
    files = []
    for name in X:
        print(name)
        file_path = os.path.join(folder_path,name)
        files.append(('file', open(file_path, 'rb')))
        
    return files


if __name__ == '__main__':

    files = load_files('data/ood_tel/test/alpha')

    #files = [('file', open(r'data/frames/vtel40.jpg', 'rb')),('file', open(r'data/ood_tel/test/alpha/wtel28.jpg', 'rb'))]
    #files = [('file', open(r'data/frames/vtel42.jpg', 'rb')),('file', open(r'data/frames/vtel259.jpg', 'rb'))]
    #files = [('file', open(r'tank.jpeg', 'rb'))]

    # query FlexServe with a batch of images (HTTP POST)
    response = requests.post("http://127.0.0.1:5050/batch", files=files)
    # query FlexServe to see names of deployed models (HTTP GET)
    #response = requests.get("http://127.0.0.1:5050/models")
    print(response.json())
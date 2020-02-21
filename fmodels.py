# Copyright <2019> Edward Verenich <verenie@clarkson.edu>
# MIT license <https://opensource.org/licenses/MIT>
import io
import json
import torch
from torchvision import transforms
from PIL import Image
import os
from torch import nn


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
MODELS_FOLDER = 'dmodels'
# specified for custom or exported, i.e. ImageNet class mappings
CLASS_NAMES = ['ALPHA', 'BETA']


# check for cuda and set gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])
# ensemble class
class Ensemble(nn.Module):
    def __init__(self, models, class_names):
        # models should be a dictionary {'name': model, 'name': model}
        super().__init__()
        self.models = models
        self.class_names = class_names

    def forward(self, inputs, batch=None):
        # dictionary of model name and output
        result = {}
        if batch:
            # concat list of inputs into tensors
            sinputs = torch.cat(inputs).to(device)
            # do stream inputs
            for name in self.models:
                outputs = self.models[name](sinputs)
                _, pred = torch.max(outputs, 1)
                ids = pred.tolist()
                # map ids to class names in an array
                value = [self.class_names[x] for x in ids]
                # result {model: ['class',...], model: ['class',...]}
                result[name] = value
            return result

        else:
            inputs = inputs.to(device)
            for name in self.models:
               outputs = self.models[name](inputs)
               _, pred = torch.max(outputs, 1)
               result[name] = self.class_names[pred.item()]
            return result
    def get_model_names(self):
        return list(self.models.keys())

# load models helper
def load_models():
    X = os.listdir(MODELS_FOLDER)
    model_list = {}
    for name in X:
        print(name)
        model_path = os.path.join(MODELS_FOLDER,name)
        model = torch.load(model_path)
        model.eval()
        model_list[name] = model
    ens = Ensemble(model_list, CLASS_NAMES)
    ens.eval()
    ens = ens.to(device)
    return ens

# create the ensemble
ens = load_models()

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return data_transforms(image).unsqueeze(0)

# single prediction
def predict(image_bytes):
    # transform bytes
    # print("PREDICT TYPEOF ----- ", len(image_bytes))
    tensor = transform_image(image_bytes)
    return ens.forward(tensor)
# prediction on batch of images
def batch_predict(image_bytes_batch):
    # transform images
    image_tensors = [transform_image(image_bytes=image_bytes) for image_bytes in image_bytes_batch]
    return ens.forward(image_tensors,batch=True)
# EXPERIMENTAL DO NOT USE
# def stream_predict(image_bytes):
#     # image_bytes contains a list of ints, convert to bytes
#     # print("SSPRED TYPEOF ----- ", type(image_bytes))
#     arr = bytearray(image_bytes)
#     # transform images
#     #image_tensors = [transform_image(image_bytes=image_bytes) for image_bytes in image_bytes_batch]
#     #tensor = torch.cat(image_tensors)
#     tensor = transform_image(arr)
#     return ens.forward(tensor)

# available models
def get_models():
    result = {}
    keys = ens.get_model_names()
    index = 0
    for k in keys:
        result[('model ' + str(index))] = k
        index += 1

    return result
    







if __name__ == '__main__':

    """X = os.listdir(MODELS_FOLDER)
    model_list = {}
    for name in X:
        print(name)
        model_path = os.path.join(MODELS_FOLDER,name)
        model = torch.load(model_path)
        model.eval()
        model_list[name] = model
    ens = Ensemble(model_list, CLASS_NAMES)
    ens = ens.to(device)"""

    with open(r"data/ood_tel/test/alpha/wtel9.jpeg", 'rb') as f:
        #tensor = transform_image(f.read())
        image_bytes = f.read()
        #image_batch = [image_bytes] * 3
        #image_tensors = [transform_image(image_bytes=image_bytes) for image_bytes in image_batch]
       
    #result = ens.forward(image_tensors, stream=True)
    result = predict(image_bytes)
    print(result)
    #print("Model list: ", ens.get_model_names())
    # test a single image
    #print(predict(image_bytes))

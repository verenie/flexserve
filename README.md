# FlexServe
Simple module to deploy PyTorch modles as REST services with flexible batching.

# deployment of models
To deploy PyTorch models using FlexServe follow the steps below:

1. Install PyTorch (FlexServe will work without CUDA but for most efficient inference it is recommended).
2. Clone the FlexServe repo or download the Python modules to your workspace.
3. Create a <b>dmodels</b> folder in your workspace and place your models in the folder.
4. Specify proper normalization values in the <i>fmodels.py</i> module:
```python
# for ImageNet based models
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
```
5. Specify class names in the order that they appear in your model's final classification layer or import mappings,
i.e. ImageNet class mappings if using publicly avialable pre-trained architecture.
```python
CLASS_NAMES = ['ALPHA', 'BETA']
```
7. Specify host ip and port in the <b>app.py</b> module if required.
6. Run the <b>app.py</b> module:
```console
user@machine:~/flexserve: python app.py
```

7. Requests can now be sent to the server at one of the endpoints, see <b>sas_client.py</b> for an example.

# Deploy example TEL models and test using example client
1. Follow steps 1-3 above to setup your workspace.
2. Go to the sample models repository at https://github.com/verenie/telmodels and download any number of models and place
them into the <i>dmodels</i> folder in your workspace.
4. Start the FlexServe module as in step 6 above. Deployed models will output ALPHA for presence of a transporter erector launcher and BETA for absense.
5. Download the data files or go to https://github.com/verenie/transfer/tree/master/data to download a compressed folder
with sample images.
6. Place the data folder in the same workspace from which you will run the <b>sas_client.py</b> module and specify the folder
path in the <b>sas_client.py</b> module:
```python
files = load_files('data/ood_tel/test/alpha')
```
7. In a separate terminal session run the <b>sas_client.py</b> module, your output should look similar to below, which represents
a call to the <i>batch</i> endpoint with two models deployed and six images sent over by the client:
```console
{'densenet169ACFEf0.pt': ['ALPHA', 'BETA', 'BETA', 'BETA', 'ALPHA', 'ALPHA'], 
'resnet34SEFEf2.pt': ['ALPHA', 'BETA', 'ALPHA', 'ALPHA', 'ALPHA', 'ALPHA']}
```

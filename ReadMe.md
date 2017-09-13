
# Read Me


This project demonstrates on how to use AML for recognizing hand written digits using CNN based neural network through Tensor Flow. **In this project, we intentionally didn't used readily available layers "*tf.layers.conv2d()*" but rather created layers using raw neural network functions available in tensorflow.**

## Prerequisites
Open command prompt from Azure Machine Learning File Menu and execute the following commands:

- pip install tensorflow
- pip install Pillow
- pip uninstall azure-cli-ml
- pip install azure-cli-ml

The **"mnistimgscore.py" and "score.py"** depends on the model and hence, user **must first run** the training script (cifair.py) in their environment before executing scoring scripts. 

## Training


Run cifair.py in a local Docker container

```
$ az ml experiment submit -c local cifair.py
```

**Download all files** in output directory to the directory outside your project and also copy conda_dependencies.yml file from aml_config directory to the same directory where the following files have been downloaded

- outputs/mnist - this folder has all model files
- outputs/tflogs - this folder has event logs for tensor board

## Scoring


Run scoring.py in a local Docker container

```
$ az ml experiment submit -c local scoring.py
```

Ensure accuracy from this run is very close to the accuracy from training.

Now run mnistimgscore.py in a local docker container

```
$ az ml experiment submit -c local mnistimgscore.py
```

Make sure the run is successful and you are able to see scored probabilities and labels for sample images

## Deployment


Before starting deployment, ensure that you have downloaded  folder **outputs/mnist** from training run and downloaded **mnistschema.json** from the outputs folder of the scoring.py run into a folder outside Azure Machine Learning Workbench project folder. 

Next step is to setup environment and publish web service. This assumes IT admin has already created modelmanagement account for you and have setup an ACS environment for you

```
#setup environment
$ az ml account modelmanagement set -n <model mgmt acct e.g. neerajteam2hosting> -g <azure resource group e.g. amlrsrcgrp2>
$ az ml env set -n <env name e.g. amlcluster> -g <azure resource group e.g. amlrsrcgrp2>

#create web service
$ az ml service create realtime -n mnistws1 -f mnistimgscore.py -m .\outputs\mnist -c .\conda_dependencies.yml -r python -l true -s .\outputs\mnistschema.json
```

If modelmanagement account and environment has not been setup, then you can create model management account and environment using the following commands:

```
$ az ml env setup -n <env name e.g. amlcluster> -g <azure resource group e.g. amlrsrcgrp2> # this will setup local environment. use option -c for cluster environment
$ az ml account modelmanagement create -n <acct name e.g. neerajteam2hosting> -l <location e.g. eastus2> -g <azure resource group e.g. amlrsrcgrp2> --sku-instances <sku count e.g. 1> --sku-name <pricing tier e.g. S1> 
```

## Consumption



Consume by calling client.py or curl. Execute the  curl command using data stored in the file provided at this [link](http://neerajkh.blob.core.windows.net/images/tmp)

```
# local docker execution
$ curl -X POST -H "Content-Type:application/json" -d @tmp http://127.0.0.1:32787/score
# remote cluster execution
$ curl -X POST -H "Content-Type:application/json" -d @tmp http://52.170.83.20:80/api/v1/service/mnistws/score
```

You can also consume Web service on sample images by executing client.py program

```
$ python client.py
	
	images/2.png :
   		scored labels  scored probabilities
	0              2                   1.0
```


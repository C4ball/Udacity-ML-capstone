# Internet Firewall Action Decision

This project is part of the Udacity Azure ML Nanodegree. In this project, we will work with theInternet Firewall Action Dataset. We will use Azure to configure a cloud-based machine learning production model and deploy it. We use Hyper Drive and Auto ML methods to develop the model. Then the model with higest accuary is retrieved (voting ensemble in this case. Accuracy: 99.83%) and deployed in cloud with Azure Container Instances(ACI) as a webservice, also by enabling the authentication. Once the model is deployed, the behaviour of the endpoint is analysed by getting a response from the service and logs are retrived at the end.



## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview

**Data Set Information:**

**Number of Features:** 12 features in total. 11 Attributes and 1 Class

**Number of Instances:** 65,532

**Class:**
Action feature is used as a class. There are 4 classes in total. These are allow, deny, drop and reset-both classes.

**Attribute Information:**

- Source Port
- Destination Port
- NAT Source Port
- NAT Destination Port
- Bytes
- Bytes Sent
- Bytes Received
- Packets
- Elapsed Time (sec)
- pkts_sent
- pkts_receiv


**Dataset info link:** https://archive.ics.uci.edu/ml/datasets/Internet+Firewall+Data
**Dataset link:** https://archive.ics.uci.edu/ml/machine-learning-databases/00542/log2.csv

### Task
In this project, we train a model to predict the Action taken by the Firewall (allow, deny, drop and reset-both)

### Access
The dataset is accessed directly from URL provided on the link above and saved as a tabular "Web file"

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
"Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality." - Font: https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml

AutoMLConfig is then defined for successfully executing the AutoML run with automl_settings and compute target , the automl_settings is defined with experiment timeout as ‘60’ minutes, task type as ‘Classification’ (Classification is a type of supervised learning in which models learn using training data, and apply those learnings to new data. Classification models is to predict which categories new data will fall into based on learnings from its training data) primary metric as ‘accuracy’, label column as ‘Action’, training_dataset to “internet-firewall-dataset”,max concurrent iterations as “5”, featurization as “auto”, early stopping enaled and model explainability enabled.

### Results
The best AutoML model obtained an accuracy of 99.832%, which may indicate an overfitted result, but considering that it is based on a software decision and the Best Hyperdrive model obtained an accuracy of 98.255%, we can assume that the AutoML model has a good result. To improve this process, we could use a cross-validation technique or adding a validation dataset not to improve our result but to make sure the model is not overfitted. 

###### Screenshot 1: AutoML RunDetails
![RunDetais_AutoML](.//step_2/1_RunDetais_AutoML.png)	


###### Screenshot 2: Best AutoML Model
![RunDetais_AutoML](.//step_2/2_Best_model_RunID_AutoML.png)	


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

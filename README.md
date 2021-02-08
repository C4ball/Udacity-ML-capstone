# Internet Firewall Action Decision

This project is part of the Udacity Azure ML Nanodegree. In this project, we will work with theInternet Firewall Action Dataset. We will use Azure to configure a cloud-based machine learning production model and deploy it. We use Hyper Drive and Auto ML methods to develop the model. Then the model with higest accuary is retrieved (voting ensemble in this case. Accuracy: 99.83%) and deployed in cloud with Azure Container Instances(ACI) as a webservice, also by enabling the authentication. Once the model is deployed, the behaviour of the endpoint is analysed by getting a response from the service and logs are retrived at the end.


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
"Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality." - Font: https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml

AutoMLConfig is then defined for successfully executing the AutoML run with automl_settings and compute target , the automl_settings is defined with experiment timeout as ‘60’ minutes, task type as ‘Classification’ (Classification is a type of supervised learning in which models learn using training data, and apply those learnings to new data. Classification models is to predict which categories new data will fall into based on learnings from its training data) primary metric as ‘accuracy’, label column as ‘Action’, training_dataset to “internet-firewall-dataset”,max concurrent iterations as “5”, featurization as “auto”, early stopping enaled and model explainability enabled.

### Results
The best AutoML model obtained an accuracy of 99.832%, which may indicate an overfitted result, but considering that it is based on a software decision and the Best Hyperdrive model obtained an accuracy of 98.255%, we can assume that the AutoML model has a good result. To improve this process, we could use a cross-validation technique or adding a validation dataset not to improve our result but to make sure the model is not overfitted. 

###### Screenshot 1: AutoML RunDetails
![RunDetais_AutoML](.//step_2/1_RunDetais_AutoML.png)	


###### Screenshot 2: Best AutoML Model
![RunDetais_AutoML](.//step_2/2_Best_model_RunID_AutoML.png)	


## Hyperparameter Tuning
"Hyperparameter tuning is accomplished by training the multiple models, using the same algorithm and training data but different hyperparameter values. The resulting model from each training run is then evaluated to determine the performance metric for which you want to optimize (for example, accuracy), and the best-performing model is selected." - Font: https://docs.microsoft.com/en-us/learn/modules/tune-hyperparameters-with-azure-machine-learning/1-introduction

Initially in the training script (train.py),the dataset (logs.csv) is retrieved from the URL (https://archive.ics.uci.edu/ml/machine-learning-databases/00542/log2.csv) provided using TabularDatasetFactory Class (Contains methods to create a tabular dataset for Azure Machine Learning). Then the data is split as train and test with the ratio of 70:30.

The classification algorithm used here is Logistic Regression. "Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression)." - Font: https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression).

Then the training (train.py) script is passed to estimator and HyperDrive configurations to predict the best model and accuracy. The HyperDrive run is executed successfully with the help of parameter sampler, policy, estimator and HyperDrive Config, before that Azure workspace,experiment and cluster is created successfully.

**Parameters:**
- "--C" : uniform(0.1,1) - (Inverse of regularization parameter)
- "--max_iter": choice(50,100,150,200) - (Maximum number of iterations to converge) 

**Early Termination Policy:**

- BanditPolicy
  - slack_factor = 0.1
  - evaluation_interval = 1
  - delay_evaluation = 5

**HyperDrive Configuration**
- Primary metric: "Accuracy"
- Max total runs: 100
- Max concurrent runs: 5


### Results
The best HyperDrive model obtained 98.255% accuracy, trained with logistic regression, Regularization force = 0.37781905471361938 and maximum iterations = 200. Considering the 99.832% accuracy of the AutoML model, the Hiperdrive model could be improved by choosing a more robust algorithm, such as Random Forest Classifier ( [Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) ) or Voting Esemble Classifier ( [Voting Ensemble Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) ).


*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?


###### Screenshot 3: Hyperdrive RunDetails
![RunDetais_AutoML](.//step_2/3_RunDetais_Hyperdrive.png)	


###### Screenshot 4: Best Hyperdrive Model
![RunDetais_AutoML](.//step_2/4_Best_model_RunID_Hyperdrive.png)	


## Model Deployment

HyperDrive’s best model accuracy = 98.255%

AutoML’s best model accuracy = 99.832%

The model with the best accuracy is deployped as per the instructions, so the AutoML's best model is deployed.

Initially, the best model is registered and it's necessary files are downloaded. Then the Environment and inference is created with the help of required conda dependencies and score.py script file which has the intialization and exit function defined for the best model and the model is deployed with ACI(Azure Container Instance) and configurations such as cpu_cores=1, memory_gb=1.

###### Screenshot 5: Model Registration
![RunDetais_AutoML](.//step_2/5_Model_Registration.png)	


###### Screenshot 6: Model Deployment
![RunDetais_AutoML](.//step_2/6_Model_Deployment.png)	


###### Screenshot 6: Model Endpoint Active
![RunDetais_AutoML](.//step_2/7_Model_Endpoint_Active.png)	

*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

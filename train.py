from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.model_selection import train_test_split
from azureml.core import Workspace, Dataset, Run


# TODO: Create TabularDataset using TabularDatasetFactory

# Dataset
# Internet Firewall Data Data Set
# Data Set Information: There are 12 features in total. Action feature is used as a class. There are 4 classes in total. These are allow, action, drop and reset-both classes.
# Attribute Information: Source Port,Destination Port,NAT Source Port,NAT Destination Port,Action,Bytes,Bytes Sent,Bytes Received,Packets,Elapsed Time (sec),pkts_sent,pkts_received
# Link: https://archive.ics.uci.edu/ml/datasets/Internet+Firewall+Data

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required

try:
    run  = Run.get_context()
    workspace = run.experiment.workspace
except:
    workspace = Workspace.from_config()

dataset = Dataset.get_by_name(workspace, name='internet-firewall-dataset')
ds = dataset.to_pandas_dataframe()
### YOUR CODE HERE ###


### YOUR CODE HERE ###a


def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    #x_df = data.to_pandas_dataframe().dropna()
    x_df = data.dropna()
    # jobs = pd.get_dummies(x_df.job, prefix="job")
    # x_df.drop("job", inplace=True, axis=1)
    # x_df = x_df.join(jobs)
    # x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    # x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    # x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    # x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    # contact = pd.get_dummies(x_df.contact, prefix="contact")
    # x_df.drop("contact", inplace=True, axis=1)
    # x_df = x_df.join(contact)
    # education = pd.get_dummies(x_df.education, prefix="education")
    # x_df.drop("education", inplace=True, axis=1)
    # x_df = x_df.join(education)
    # x_df["month"] = x_df.month.map(months)
    # x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    # x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("Action")#.apply(lambda s: 1 if s == "yes" else 0)
    
    return x_df,y_df

def main():
    
    x, y = clean_data(ds)
    # TODO: Split data into train and test sets.
    x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
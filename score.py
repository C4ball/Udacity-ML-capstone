import json
import pandas as pd
import os
import joblib, pickle
from azureml.core import Model
from sklearn.externals import joblib
import azureml.train.automl

def init():
    global daone
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    #model_path = Model.get_model_path('best-automl.pkl')
    daone = joblib.load(model_path)

def run(data):
    try:
        
        #data = json.loads(data)
        #df = pd.DataFrame([data['data']])
        #result = model.predict(df)
        #info = {
            #"input": data,
            #"output": result.tolist()
            #}
        
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = daone.predict(data)
        
        #trynn = json.loads(data)
        #data = pd.DataFrame(trynn['data'])
        #result = daone.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
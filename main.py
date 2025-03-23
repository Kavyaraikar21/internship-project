import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle


def get_clean_data():
  data = pd.read_csv("data.csv")

  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

data1 = pd.read_csv("data.csv")
pd.set_option('display.max_columns', None)  # Ensures all columns are visible
print(data1.head())
print("Shape of the dataset:", data1.shape)
  


def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data 80 20
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  # train the model
  model = LogisticRegression()
  model.fit(X_train, y_train)
  
  # test model
  y_pred = model.predict(X_test)
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler

def main():
  data = get_clean_data()

  model, scaler = create_model(data)

  with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

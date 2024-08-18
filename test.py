import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score                          
import streamlit as st

# loading the dataset to a Pandas DataFrame
data = pd.read_csv("C:/Users/swathi/credit card project/CreditCard.csv")
        
# separating the data for analysis
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

print(legit.shape)
print(fraud.shape)

legit.Amount.describe()
fraud.Amount.describe()

data.groupby('Class').mean()

legit_sample = legit.sample(n=492)

print(legit_sample)

new_data = pd.concat([legit_sample, fraud], axis=0)

new_data['Class'].value_counts()

new_data.groupby('Class').mean()

X = new_data.drop(columns='Class', axis=1)
Y = new_data['Class']
print(X)

print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter all required features values')
input_df_splited = input_df.replace('\t', ',').split(',')

submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.asarray(input_df_splited, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1, -1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")

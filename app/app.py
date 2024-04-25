# import pickle
# import streamlit as st
# import pandas as pd
# from PIL import Image
# import os
# import sklearn
# from joblib import load




# current_dir = os.path.dirname(__file__)
# models_folder = os.path.join(current_dir, '..', 'models')
# model_file = os.path.join(models_folder, 'Gradient Boosting_model.joblib')
# encoded_columns_file = os.path.join(models_folder, 'encoded_columns.pkl')
# model = load(model_file)

# # with open(model_file, 'rb') as f_in:
# #     model = pickle.load(f_in)



# def vectorize_input(user_input, encoded_columns):
#     # Convert user input dictionary to DataFrame
#     input_df = pd.DataFrame([user_input])

#     # Ensure that column names match the encoded columns
#     for column in encoded_columns:
#         if column not in input_df.columns:
#             input_df[column] = 0

#     # Perform one-hot encoding for categorical variables
#     input_df = pd.get_dummies(input_df, columns=encoded_columns)

#     return input_df

# def vectorize_input_batch(data, encoded_columns):
#     # Ensure that column names match the encoded columns
#     for column in encoded_columns:
#         if column not in data.columns:
#             data[column] = 0

#     # Perform one-hot encoding for batch data
#     data = pd.get_dummies(data, columns=encoded_columns)

#     return data


# def main():
#     image = Image.open('images/icone.png')
#     image2 = Image.open('images/image.png')
#     st.image(image, use_column_width=False)
#     add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
#     st.sidebar.info('This app is created to predict Customer Churn')
#     st.sidebar.image(image2)
#     st.title("Predicting Customer Churn")
#     encoded_columns_file = os.path.join(models_folder, 'encoded_columns.pkl')
#     with open(encoded_columns_file, 'rb') as f:
#         encoded_columns = pickle.load(f)

#     if add_selectbox == 'Online':
#         gender = st.selectbox('Gender:', ['male', 'female'])
#         seniorcitizen = st.selectbox('Customer is a senior citizen:', [0, 1])
#         partner = st.selectbox('Customer has a partner:', ['yes', 'no'])
#         dependents = st.selectbox('Customer has dependents:', ['yes', 'no'])
#         phoneservice = st.selectbox('Customer has phone service:', ['yes', 'no'])
#         multiplelines = st.selectbox('Customer has multiple lines:', ['yes', 'no', 'no_phone_service'])
#         internetservice = st.selectbox('Customer has internet service:', ['dsl', 'no', 'fiber_optic'])
#         onlinesecurity = st.selectbox('Customer has online security:', ['yes', 'no', 'no_internet_service'])
#         onlinebackup = st.selectbox('Customer has online backup:', ['yes', 'no', 'no_internet_service'])
#         deviceprotection = st.selectbox('Customer has device protection:', ['yes', 'no', 'no_internet_service'])
#         techsupport = st.selectbox('Customer has tech support:', ['yes', 'no', 'no_internet_service'])
#         streamingtv = st.selectbox('Customer has streaming TV:', ['yes', 'no', 'no_internet_service'])
#         streamingmovies = st.selectbox('Customer has streaming movies:', ['yes', 'no', 'no_internet_service'])
#         contract = st.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
#         paperlessbilling = st.selectbox('Customer has paperless billing:', ['yes', 'no'])
#         paymentmethod = st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
#         tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
#         monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
#         totalcharges = tenure * monthlycharges
#         input_dict = {
#             "gender": gender,
#             "seniorcitizen": seniorcitizen,
#             "partner": partner,
#             "dependents": dependents,
#             "phoneservice": phoneservice,
#             "multiplelines": multiplelines,
#             "internetservice": internetservice,
#             "onlinesecurity": onlinesecurity,
#             "onlinebackup": onlinebackup,
#             "deviceprotection": deviceprotection,
#             "techsupport": techsupport,
#             "streamingtv": streamingtv,
#             "streamingmovies": streamingmovies,
#             "contract": contract,
#             "paperlessbilling": paperlessbilling,
#             "paymentmethod": paymentmethod,
#             "tenure": tenure,
#             "monthlycharges": monthlycharges,
#             "totalcharges": totalcharges
#         }

#         if st.button("Predict"):
#             X = vectorize_input(input_dict, encoded_columns)
#             X_array = X.values
#             print(X_array)

#             y_pred = model.predict(X_array)
#             churn = y_pred
#             st.success('Churn:', churn)

            

#     if add_selectbox == 'Batch':
#         file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
#         if file_upload is not None:
#             data = pd.read_csv(file_upload)
#             encoded_columns_file = os.path.join(models_folder, 'encoded_columns.pkl')
#             with open(encoded_columns_file, 'rb') as f:
#                 encoded_columns = pickle.load(f)
#             X = vectorize_input_batch(data, encoded_columns)
#             y_pred = model.predict(X)[:, 1]
#             # churn = y_pred >= 0.5
#             # churn = [bool(val) for val in churn]
#             st.write(y_pred)

# if __name__ == '__main__':
#     main()


import streamlit as st
import os
import pickle
import joblib
import pandas as pd
from PIL import Image

# Load the encoded columns
encoded_columns_file = r"C:\Users\pc\OneDrive\Desktop\Customer_Churn\models\encoded_columns.pkl"
with open(encoded_columns_file, 'rb') as f:
    encoded_columns = pickle.load(f)

# Load the model
model_file = r"C:\Users\pc\OneDrive\Desktop\Customer_Churn\models\Gradient Boosting_model.joblib"  # Change this to your model file path
model = joblib.load(model_file)

def encode_input(input_dict):
    # Create DataFrame from input dictionary
    input_df = pd.DataFrame([input_dict])
    # One-hot encode categorical columns
    encoded_input = pd.get_dummies(input_df, columns=encoded_columns, drop_first=True)
    return encoded_input

def predict_churn(encoded_input):
    prediction = model.predict(encoded_input)
    return prediction[0]

def main():
    image = Image.open('images/icone.png')
    image2 = Image.open('images/image.png')
    st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")

    if add_selectbox == 'Online':
        st.subheader("Online Prediction")
        gender = st.selectbox('Gender:', ['male', 'female'])
        seniorcitizen = st.selectbox('Customer is a senior citizen:', [0, 1])
        partner = st.selectbox('Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox('Customer has dependents:', ['yes', 'no'])
        phoneservice = st.selectbox('Customer has phone service:', ['yes', 'no'])
        multiplelines = st.selectbox('Customer has multiple lines:', ['yes', 'no', 'no_phone_service'])
        internetservice = st.selectbox('Customer has internet service:', ['dsl', 'no', 'fiber_optic'])
        onlinesecurity = st.selectbox('Customer has online security:', ['yes', 'no', 'no_internet_service'])
        onlinebackup = st.selectbox('Customer has online backup:', ['yes', 'no', 'no_internet_service'])
        deviceprotection = st.selectbox('Customer has device protection:', ['yes', 'no', 'no_internet_service'])
        techsupport = st.selectbox('Customer has tech support:', ['yes', 'no', 'no_internet_service'])
        streamingtv = st.selectbox('Customer has streaming TV:', ['yes', 'no', 'no_internet_service'])
        streamingmovies = st.selectbox('Customer has streaming movies:', ['yes', 'no', 'no_internet_service'])
        contract = st.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox('Customer has paperless billing:', ['yes', 'no'])
        paymentmethod = st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
        monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure * monthlycharges
        input_dict = {
            "gender": gender,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "tenure": tenure,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }
        encoded_input = encode_input(input_dict)
        prediction = predict_churn(encoded_input)
        st.write("Predicted Churn Probability:", prediction)

    elif add_selectbox == 'Batch':
        st.subheader("Batch Prediction")
        st.write("Upload a CSV file containing the customer data:")
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file is not None:
            batch_data = pd.read_csv(csv_file)
            # Perform batch prediction
            encoded_batch_data = encode_input(batch_data)
            predictions = model.predict(encoded_batch_data)
            batch_data['Predicted Churn Probability'] = predictions
            st.write(batch_data)

if __name__ == "__main__":
    main()


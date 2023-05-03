import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from datetime import datetime
# import joblib

# Load the data
data = pd.read_csv('biofuel_data.csv')

# Split the data into input features (X) and target variable (y)
X = data.drop(['Biofuel Production'], axis=1)
y = data['Biofuel Production']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the preprocessing pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# Define the preprocessing pipeline for categorical features
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the preprocessing pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, ['Cellulose',
     'Hemicellulose', 'Lignin', 'Ash', 'Extractives']),
    ('cat', cat_pipeline, ['Species'])
])

# Define the Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Combine the preprocessing pipeline and model in a single pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', rf)
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)


st.markdown('''<center><h1>BIOFUEL PREDICTION</h1><center></br></br>''',
            unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)
with col1:
    cellulose = st.text_input("Cellulose:", placeholder="30.0 - 38.5")
    hemicellulose = st.text_input("Hemicellulose:", placeholder="18.0 - 24.5")
with col2:
    lignin = st.text_input("Lignin:", placeholder="21.5 - 26.5")
    ash = st.text_input("Ash:", placeholder="3.9 - 5.6")
with col3:
    extractives = st.text_input("Extractives:", placeholder="2.1 - 3.7")
    species = st.selectbox("Select the species type:", ("A", "B", "C", "D"))

try:
    user_input = pd.DataFrame({'Cellulose': [cellulose], 'Hemicellulose': [hemicellulose], 'Lignin': [
        lignin], 'Ash': [ash], 'Extractives': [extractives], 'Species': [species]})
    if st.button("Predict"):
        # Make prediction on user input
        user_prediction = pipeline.predict(user_input)

        st.write(
            f'##### Predicted Biofuel Production (L/kg): {round(user_prediction[0],4)}%')
        Biofuel = round(user_prediction[0], 4)
        df = user_input
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        df["Biofuel"] = Biofuel
        df["date_string"] = date_string
        output_path = 'database/predicted.csv'
        df.to_csv(output_path, mode='a', index=False,
                  header=not os.path.exists(output_path))
        with st.container():
            st.vega_lite_chart(data, {
                'mark': {'type': 'circle', 'tooltip': True},
                'encoding': {
                    'x': {'field': 'Cellulose', 'type': 'quantitative'},
                    'y': {'field': 'Hemicellulose', 'type': 'quantitative'},
                    'size': {'field': 'Lignin', 'type': 'quantitative'},
                    'color': {'field': 'Biofuel Production', 'type': 'quantitative'},
                },
            })

except Exception as e:
    pass

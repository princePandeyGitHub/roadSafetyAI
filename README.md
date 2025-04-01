# roadSafetyAI
Accident prevention system


This project aims to predict the severity of accidents based on various features related to the accident. The goal is to create a machine learning model that can analyze accident-related data and predict the severity of accidents, such as whether they were mild, severe, or fatal.

Below is a detailed breakdown of each step involved in the project.

1. Problem Understanding:
The primary goal of this project is to predict the severity of accidents based on various input features. These features may include:

Location-based features: Longitude, Latitude

Accident-related features: Number of vehicles involved, Speed limit

Time-based features: Day of the week, Time of the day

Road and weather conditions: Road type, Junction control, Pedestrian crossing control, Light conditions, Weather conditions, and Road surface conditions

Geographical data: Urban or rural area

The output or target variable is Accident Severity, which can be classified into different categories like:

Severity 1: Minor

Severity 2: Serious

Severity 3: Fatal

The goal is to build a classification model that will take the input features and output a severity category for an accident.

2. Loading and Preprocessing Data:
a. Loading Data:
The data is loaded using the pandas.read_csv() method. This function reads the CSV file that contains the accident data into a pandas DataFrame.

python
Copy
Edit
df = pd.read_csv("your_dataset.csv")
You would replace "your_dataset.csv" with the actual file path of your dataset.

b. Handling Missing Values:
Data cleaning is an essential step in any machine learning project. Missing or null values can lead to incorrect model training and poor performance.

Target Column (Accident_Severity): We drop rows where the target variable (Accident_Severity) is missing.

python
Copy
Edit
df = df.dropna(subset=['Accident_Severity'])
Other Columns: For numerical features with missing values, we fill them with their mean to ensure that there are no gaps in the data.

python
Copy
Edit
df = df.fillna(df.mean())
c. Feature Selection:
The next step is to select relevant features that will be used for predicting the accident severity. Based on the dataset, we choose features that are likely to have an impact on accident severity:

python
Copy
Edit
features = [
    'Longitude', 'Latitude', 'Number_of_Vehicles', 'Speed_limit', 
    'Day_of_Week', 'Time', 'Road_Type', 'Junction_Control', 
    'Pedestrian_Crossing-Human_Control', 'Light_Conditions', 'Weather_Conditions', 
    'Road_Surface_Conditions', 'Urban_or_Rural_Area'
]
The target variable is Accident_Severity.

3. Feature Engineering and Preprocessing:
a. Handling Categorical Variables:
Some features like Weather_Conditions, Road_Surface_Conditions, and Junction_Control may be categorical (text-based values). These categorical variables need to be converted into numerical form using one-hot encoding (dummy variables) so that they can be used in machine learning models.

python
Copy
Edit
df = pd.get_dummies(df, columns=['Weather_Conditions', 'Road_Surface_Conditions', 'Junction_Control', 'Time'])
This converts categorical columns into multiple binary columns (0 or 1) representing different categories.

b. Feature Scaling:
Machine learning models work better when the features are on a similar scale. Features like longitude, latitude, speed limit, etc., may have different ranges, so we use StandardScaler to scale the features to a standard normal distribution (mean = 0, standard deviation = 1).

python
Copy
Edit
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
4. Preparing Data for Training:
After preprocessing the data, we need to separate the features (X) and the target (y). We use the following:

python
Copy
Edit
X = df[features]
y = df[target]
Next, we split the data into training and test sets. Typically, 80% of the data is used for training, and 20% is used for testing the model.

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
5. Model Training:
Now that the data is prepared, we can start training a machine learning model. For this task, we use a Random Forest Classifier, which is a robust and widely used ensemble learning method for classification tasks.

python
Copy
Edit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
n_estimators=100 specifies the number of decision trees in the random forest.

random_state=42 ensures reproducibility of results.

6. Model Evaluation:
After training the model, we evaluate its performance on the test data using the accuracy score. The accuracy score measures the proportion of correctly predicted accident severities.

python
Copy
Edit
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
This gives us an idea of how well the model generalizes to new, unseen data.

7. Saving the Model:
After training and evaluating the model, we save the trained model to a file using pickle so that it can be reused for future predictions without retraining.

python
Copy
Edit
with open('accident_model.pkl', 'wb') as f:
    pickle.dump(model, f)
Now, the model is stored in the file accident_model.pkl, and you can load it anytime to make predictions.

8. Correlation Heatmap:
A correlation heatmap helps visualize the relationships between different numerical features. It's a useful tool to explore how various features correlate with each other, which can help in feature selection and understanding the data better.

python
Copy
Edit
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
9. Prediction Function:
Now, the project includes a prediction function that allows the user to input their own data (for example, accident details) and predict the accident severity based on the trained model.

python
Copy
Edit
def predict_accident(features):
    with open('accident_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    scaled_features = scaler.transform([features])
    
    prediction = model.predict(scaled_features)
    
    return prediction[0]
The user inputs their own data (for example, longitude, latitude, speed limit, etc.), which is then preprocessed and scaled to match the format expected by the model.

The model is loaded, and the prediction is made.

10. Example Prediction:
Finally, we show an example of how a user would use the predict_accident() function to make a prediction. The features would correspond to the same format expected by the model.

python
Copy
Edit
user_input = [
    78.61039332, 14.72402585, 2, 30, 3, 17, 0, 1, 1, 1, 0, 0
    # [Longitude, Latitude, Number_of_Vehicles, Speed_limit, Day_of_Week, Time, Road_Type, Junction_Control, Pedestrian_Crossing, Light_Conditions, Weather_Conditions, Urban_or_Rural_Area]
]

predicted_severity = predict_accident(user_input)
print(f"Predicted Accident Severity: {predicted_severity}")
This gives the user the predicted accident severity based on the features they provide.

Conclusion:
This project demonstrates how to:

Clean and preprocess accident data.

Select and transform features.

Train a machine learning model (Random Forest Classifier) to predict accident severity.

Save and reload the trained model.

Make predictions using user inputs.

The goal of this project is to predict the severity of an accident based on multiple features such as location, weather, road conditions, and more. You can improve this model by experimenting with other classifiers, fine-tuning hyperparameters, and handling more complex datasets.

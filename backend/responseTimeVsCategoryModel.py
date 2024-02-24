import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load ticket data
ticket_data = pd.read_csv('../data/ticket_data.csv')
# Preprocessing
# Convert timestamps to datetime objects
ticket_data['ticket_creation_timestamp'] = pd.to_datetime(ticket_data['ticket_creation_timestamp'])
ticket_data['agent_response_timestamp'] = pd.to_datetime(ticket_data['agent_response_timestamp'])
ticket_data['ticket_resolution_timestamp'] = pd.to_datetime(ticket_data['ticket_resolution_timestamp'])
# Calculate response time
ticket_data['response_time'] = (
            ticket_data['agent_response_timestamp'] - ticket_data['ticket_creation_timestamp']).dt.total_seconds()
encoder = OrdinalEncoder()
ticket_data['ticket_priority'] = encoder.fit_transform(ticket_data[['ticket_priority']])
ticket_data['ticket_category'] = encoder.fit_transform(ticket_data[['ticket_category']])
# Feature selection
features = ['ticket_priority', 'ticket_category', 'response_time']
X = ticket_data[features]
y = ticket_data['response_time']  # Target variable
# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
import joblib

joblib.dump(model, 'random_forest_model.joblib')
# Predictions
# Use the last 30 days of data for prediction
last_date = ticket_data['ticket_creation_timestamp'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
# future_features = pd.DataFrame({'ticket_priority': 'High', 'ticket_category': 'General', 'response_time': 0}, index=future_dates)
priority_encoder = LabelEncoder()
category_encoder = LabelEncoder()

# Fit your label encoders with your training data categories and priorities
# This is just an example, in practice, use the categories and priorities from your training data
priority_encoder.fit(['Low', 'Medium', 'High'])  # Replace with the actual priorities from your dataset
category_encoder.fit(['Technical', 'General'])  # Replace with the actual categories from your dataset

# Transform the 'High' priority and 'General' category using the fitted label encoders
encoded_priority = priority_encoder.transform(['Low'])[0]  # Get the encoded value for 'High'
encoded_category = category_encoder.transform(['General'])[0]  # Get the encoded value for 'General'

# Generate the DataFrame for future features with encoded values
future_features = pd.DataFrame({
    'ticket_priority': encoded_priority,
    'ticket_category': encoded_category,
    'response_time': 5  # Assuming a placeholder value of 0 for response_time
}, index=future_dates)
future_predictions = model.predict(future_features)
# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_predictions, marker='o', linestyle='-', color='b')
plt.title('Predicted Ticket Resolution Time for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Resolution Time (seconds)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
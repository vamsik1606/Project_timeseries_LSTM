import os
import subprocess
import getpass
import sys
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

import sqlalchemy as sa

from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from torch.utils.data import DataLoader, TensorDataset
#this is to connect to the internal clusters, for data sources and data processors
s3_connector = S3Connector()
engine_creator = EngineCreator()
test_fleet_engine = engine_creator.get_engine('')
spark = build_spark_session(app_name='',
                            size='dynamic',
                            s3_connector = s3_connector)

# Load the parquet file
df = pd.read_parquet("")
#removing unnecassary columns
remove_columns = ['']
df = df.drop(columns = remove_columns)
list(df)
df = df.fillna(0)
#look for unique vehicles
df['grouped_vehicle'].unique()

#split the vehicles into test and train dataframes

test_df = df[df['grouped_vehicle'].isin(['vehicle_1','vehicle_2'])]
train_df = df[df['grouped_vehicle'].isin(['vehicle_4', 'vehicle_3'])]

unique_sessions = train_df['sessionid'].unique()
unique_sessions_test = test_df['sessionid'].unique()
len(unique_sessions),len(unique_sessions_test)



################# Define LSTM Model ####################################
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1, :])
        return self.softmax(predictions)

# Generate dummy sequential data
num_samples = 1000
sequence_length = 10
input_size = 1
num_classes = 3

X = torch.randn(num_samples, sequence_length, input_size)
y = torch.randint(0, num_classes, (num_samples,))
y = torch.nn.functional.one_hot(y, num_classes).float()

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = LSTM(input_size=input_size, hidden_size=50, output_size=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Predict on new data
sample_input = torch.randn(1, sequence_length, input_size)
prediction = model(sample_input)
predicted_class = torch.argmax(prediction, dim=1).item()
print(f"Predicted Class: {predicted_class}")
###########################Training############################################
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert time to numerical format
train_df['time'] = pd.to_numeric(train_df['time'], errors='coerce')
test_df['time'] = pd.to_numeric(test_df['time'], errors='coerce')

# Data preprocessing
train_df = train_df.sort_values(by=['sessionid', 'time'])
test_df = test_df.sort_values(by=['sessionid', 'time'])

train_sessions = train_df['sessionid'].unique()
test_sessions = test_df['sessionid'].unique()

# Ensure all feature columns are numeric
feature_columns = train_df.drop(columns=['auto', 'sessionid']).columns
train_df[feature_columns] = train_df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
test_df[feature_columns] = test_df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Scaling features
scaler = MinMaxScaler()
train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
test_df[feature_columns] = scaler.transform(test_df[feature_columns])

# Extract unique target values
unique_targets = sorted(train_df['auto'].unique())
num_classes = len(unique_targets)

# Create a mapping from values to class indices
target_to_index = {val: idx for idx, val in enumerate(unique_targets)}
index_to_target = {idx: val for idx, val in enumerate(unique_targets)}

# Convert target values to indices
train_df['auto'] = train_df['auto'].map(target_to_index)
test_df['auto'] = test_df['auto'].map(target_to_index)

# Model Definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Prepare data
x_train = train_df.drop(columns=['auto', 'sessionid'])
t_train = train_df['auto']

x_train = x_train.apply(pd.to_numeric, errors='coerce').fillna(0)
t_train = pd.to_numeric(t_train, errors='coerce').fillna(0)

input_dim = x_train.shape[1]
model = LSTM(input_size=input_dim, hidden_size=32, num_classes=num_classes).to(device)

# Use CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Plot actual vs. predicted values for a session
def plot_results(session, actual_vals, pred_vals):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(actual_vals)), actual_vals, label='Actual', color='blue', alpha=0.6)
    plt.scatter(range(len(pred_vals)), pred_vals, label='Predicted', color='red', alpha=0.6)
    
    plt.xlabel('Time Step')
    plt.ylabel('auto Value')
    plt.title(f'Session {session} - Actual vs. Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

# Training Loop (Modified to include plotting)
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    for session in train_sessions:
        session_data = train_df[train_df['sessionid'] == session]
        x_session = session_data.drop(columns=['auto', 'sessionid']).values
        t_session = session_data['auto'].values

        if len(x_session) == 0 or len(t_session) == 0:
            print(f"Skipping empty session {session}")
            continue

        x_session = torch.tensor(x_session, dtype=torch.float32).to(device)
        t_session = torch.tensor(t_session, dtype=torch.long).to(device)

        pred_vals = []
        actual_vals = []
        losses = []

        batch_size = len(x_session)
        for i in range(0, len(x_session), batch_size):
            x_batch = x_session[i:i+batch_size].unsqueeze(1).to(device)
            t_batch = t_session[i:i+batch_size].to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, t_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            predicted_indices = torch.argmax(output, dim=1)
            predicted_values = [index_to_target[idx.item()] for idx in predicted_indices]

            pred_vals.extend(predicted_values)
            actual_vals.extend([index_to_target[idx.item()] for idx in t_batch])
            losses.append(loss.item())

        # Print table of actual vs predicted values
        #results_df = pd.DataFrame({'Actual': actual_vals, 'Predicted': pred_vals})
        #print(f"Session {session} - Actual vs Predicted:")
        #print(results_df)

        # Plot results for this session
        plot_results(session, actual_vals, pred_vals)

        print(f'Session {session} Complete. Final Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Training complete! Model saved to model.pth")


###############Testing#####################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode

test_sessions = test_df['sessionid'].unique()

def plot_results(session, actual_vals, pred_vals):
    """Plots actual vs. predicted values for a given session."""
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(actual_vals)), actual_vals, label='Actual', color='blue', alpha=0.6, s=18)
    plt.scatter(range(len(pred_vals)), pred_vals, label='Predicted', color='red', alpha=0.6, s=5)
    plt.xlabel('Time Step')
    plt.ylabel('auto Value')
    plt.title(f'Test Session {session} - Actual vs. Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

all_actual = []
all_predicted = []

for session in test_sessions:
    session_data = test_df[test_df['sessionid'] == session]
    x_session = session_data.drop(columns=['auto', 'sessionid']).values
    t_session = session_data['auto'].values

    if len(x_session) == 0:
        print(f"Skipping empty session {session}")
        continue

    x_session_tensor = torch.tensor(x_session, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(x_session_tensor.unsqueeze(1))  # Ensure batch format
        predicted_indices = torch.argmax(output, dim=1).cpu().numpy()

    predicted_values = [index_to_target[idx] for idx in predicted_indices]
    actual_values = [index_to_target[target_to_index[val]] for val in t_session]

    all_actual.extend(actual_values)
    all_predicted.extend(predicted_values)

    results_df = pd.DataFrame({'Actual': actual_values, 'Predicted': predicted_values})
    print(f"Test Session {session} - Actual vs Predicted:")
    print(results_df)
    
    plot_results(session, actual_values, predicted_values)

# Compute evaluation metrics
overall_accuracy = accuracy_score(all_actual, all_predicted)
overall_precision = precision_score(all_actual, all_predicted, average='weighted')
overall_recall = recall_score(all_actual, all_predicted, average='weighted')
overall_f1 = f1_score(all_actual, all_predicted, average='weighted')

print("\nOverall Model Performance:")
print(f"Accuracy: {overall_accuracy:.4f}")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall: {overall_recall:.4f}")
print(f"F1 Score: {overall_f1:.4f}")

print("Testing complete!")

#########corelation matrix###########
# Exclude non-numeric columns
numeric_df = train_df.drop(columns=['sessionid'])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Extract correlation with the output variable
target_correlation = correlation_matrix['auto'].drop('auto')

# Plot heatmap
plt.figure(figsize=(10, 15))
sns.heatmap(target_correlation.to_frame(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation with Output Variable")
plt.xlabel("Output Variable")
plt.ylabel("Input Features")
plt.show()
###############Confusion matrix##########
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
from sklearn.metrics import accuracy_score

# Convert predicted and actual values to numerical indices
actual_indices = [target_to_index[val] for val in actual_values]
predicted_indices = [target_to_index[val] for val in predicted_values]

# Compute accuracy
accuracy = accuracy_score(actual_indices, predicted_indices)
print(f"Model Accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(actual_indices, predicted_indices)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=target_to_index.keys(), yticklabels=target_to_index.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

###save the model###
torch.save(model.state_dict(), 'model.pth')
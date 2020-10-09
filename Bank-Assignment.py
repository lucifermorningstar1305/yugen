#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style
from jupyterthemes import jtplot 
from joblib import dump
jtplot.style(theme="monokai", context="notebook", ticks=True)


# In[2]:


df = pd.read_csv("bank-full.csv", delimiter=";")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


# Get all numeric columns 
numeric_type_fields = df.select_dtypes(exclude=["object"]).columns
print(Fore.YELLOW, "Numeric Values in ", Style.RESET_ALL, numeric_type_fields)

# Get all string columns
object_type_fields = df.select_dtypes(exclude=['int64']).columns
print(Fore.BLUE, "String values in ", Style.RESET_ALL, object_type_fields)


# In[6]:


# List all categorical values for the object columns
def get_categorical_info():
    colors = [Fore.YELLOW, Fore.BLUE, Fore.RED]

    for idx, cols in enumerate(object_type_fields):

        if idx >= 3 :
            idx = idx % 3

        print(colors[idx], f"{cols}", Style.RESET_ALL, f" --> {df[cols].unique()}")
    
get_categorical_info()


# # Observations of Categorical Values:
# 
# * Except the `month` and `education` every other field are nominal data.
# * The `job` field has similar categorical values which are named either differently or abbreviated such 'management' and 'mgmt', 'admin' and 'admin.'
# * Since these data are in object format they need to be converted to numeric data for model training and others.

# In[7]:


# Cleaning part 1
df["job"].loc[df['job'] == "admin."] = "admin"
df["job"].loc[df["job"] == "mgmt"] = "management"

get_categorical_info()


# In[8]:


df.loc[df["education"] == "unknown"]


# In[9]:


df.loc[(df["education"] == "unknown") & (df["job"] == "student")]


# In[10]:


# Encoding data
jobs = {"management":0, "technician" : 1, "entrepreneur": 2, "blue-collar": 3, 
       "unknown" : 4, "retired" : 5, "admin" : 6, "services" : 7, "self-employed" : 8,
       "unemployed" : 9, "housemaid" : 10, "student": 11} # Nominal field

marital = {"single" : 0, "married": 1, "divorced": 2} # Nominal field

educational = {"unknown":0, "primary" : 1, "secondary" : 2, "tertiary": 3} # Ordinal field

yes_no_type_fields = {"no" : 0, "yes" : 1}

contact = {"unknown" : 0, "cellular" : 1, "telephone" : 2} # Nominal field

month = {"jan":0, "feb":1, "mar":2, "apr":3, "may":4, "jun":5, "jul":6,
        "aug":7, "sep":8, "oct":9, "nov":10, "dec":11} # Ordinal field

poutcome = {"unknown":0, "failure":1, "other":2, "success": 3} # Nominal field


# Apply these encoding to the categorical fields

df["job"] = df["job"].apply(lambda x: jobs[x])
df["marital"] = df["marital"].apply(lambda x: marital[x])
df["education"] = df["education"].apply(lambda x: educational[x])
df["default"] = df["default"].apply(lambda x: yes_no_type_fields[x])
df["housing"] = df["housing"].apply(lambda x: yes_no_type_fields[x])
df["loan"] = df["loan"].apply(lambda x: yes_no_type_fields[x])
df["y"] = df["y"].apply(lambda x: yes_no_type_fields[x])
df["month"] = df["month"].apply(lambda x: month[x])
df["contact"] = df["contact"].apply(lambda x: contact[x])
df["poutcome"] = df["poutcome"].apply(lambda x: poutcome[x])

# Let's look at the DataFrame now after applying these encoding
df.head()


# # Target Distribution

# In[11]:


plt.figure(figsize=(15, 10))
sns.countplot(x="y", data=df)


# # Outlier Analysis
# 
# For the Outlier Analysis I am applying the IQR rule the formula for which is 
# 
# $$
# \begin{equation}
# IQR := Q3 - Q1 \\
# LB := Q1 - (1.5 \times IQR) \\
# UB := Q3 + (1.5 \times IQR)
# \end{equation}
# $$
# 
# The Other workaround is to use Z-Scores whose formula is :
# 
# $$
# \begin{equation}
# Z := \frac{x - \mu}{\sigma}
# \end{equation}
# $$
# 
# **Note** : IQR is non-parameteric while z-score is parametric and Z-Score assumes that the data is a Normal Distribution

# ### Outliers in `age` field

# In[12]:


q1_age = df["age"].quantile(0.25)
q3_age = df["age"].quantile(0.75)
iqr_age = q3_age - q1_age
lb_age = q1_age - (1.5 * iqr_age)
ub_age = q3_age + (1.5 * iqr_age)

print(f"Any values <{lb_age} : {df.loc[df['age'] < lb_age, 'age'].unique()}, len: {len(df.loc[df['age'] < lb_age, 'age'].unique())}")
print(f"Any values >{ub_age} : {df.loc[df['age'] > ub_age, 'age'].unique()}, len : {len(df.loc[df['age'] > ub_age, 'age'].unique())}")


# In[13]:


df.boxplot(column="age")


# ### Outliers in `balance` field

# In[14]:


q1_balance = df["balance"].quantile(0.25)
q3_balance = df["balance"].quantile(0.75)
iqr_balance = q3_balance - q1_balance
lb_balance = q1_balance - (1.5 * iqr_balance)
ub_balance = q3_balance + (1.5 * iqr_balance)

print(f"Any values < {lb_balance} : {df.loc[df['balance'] < lb_balance, 'balance'].unique()}, len : {len(df.loc[df['balance'] < lb_balance])}")
print(f"Any values >{ub_balance} : {df.loc[df['balance'] > ub_balance, 'balance'].unique()}, len: {len(df.loc[df['balance'] > ub_balance])}")


# In[15]:


df.boxplot(column="balance")


# ### Outliers in `day` field

# In[16]:


q1_day = df["day"].quantile(0.25)
q3_day = df["day"].quantile(0.75)
iqr_day = q3_day - q1_day
lb_day = q1_day - (1.5 * iqr_day)
ub_day = q3_day + (1.5 * iqr_day)
print(f"Any values <{lb_day}:{df.loc[df['day'] < lb_day, 'day'].unique()}, len:{len(df.loc[df['day'] < lb_day])}")
print(f"Any values >{ub_day}:{df.loc[df['day'] > ub_day, 'day'].unique()}, len:{len(df.loc[df['day'] > ub_day])}")


# In[17]:


df.boxplot(column="day")


# ### Outliers in `duration`

# In[18]:


q1_duration = df["duration"].quantile(0.25)
q3_duration = df["duration"].quantile(0.75)

iqr_duration = q3_duration - q1_duration
lb_duration = q1_duration - (1.5 * iqr_duration)
ub_duration = q3_duration + (1.5 * iqr_duration)

print(f"Any values <{lb_duration}:{df.loc[df['duration'] < lb_duration, 'duration'].unique()}, len : {len(df.loc[df['duration'] < lb_duration, 'duration'].unique())}")
print(f"Any values >{ub_duration}:{df.loc[df['duration'] > ub_duration, 'duration'].unique()}, len : {len(df.loc[df['duration'] > ub_duration, 'duration'].unique())}")


# In[19]:


df.boxplot(column="duration")


# ### Outliers in `campaign`

# In[20]:


q1_campaign = df["campaign"].quantile(0.25)
q3_campaign = df["campaign"].quantile(0.75)
iqr_campaign = q3_campaign - q1_campaign

lb_campaign = q1_campaign - (1.5 * iqr_campaign)
ub_campaign = q3_campaign + (1.5 * iqr_campaign)

print(f"Any values <{lb_campaign} : {df.loc[df['campaign'] < lb_campaign, 'campaign'].unique()}, len:{len(df.loc[df['campaign'] < lb_campaign, 'campaign'].unique())}")
print(f"Any values >{ub_campaign} : {df.loc[df['campaign'] > ub_campaign, 'campaign'].unique()}, len:{len(df.loc[df['campaign'] > ub_campaign, 'campaign'].unique())}")


# In[21]:


df.boxplot(column="campaign")


# ### Outliers in `pdays`

# In[22]:


q1_pdays = df["pdays"].quantile(0.25)
q3_pdays = df["pdays"].quantile(0.75)

iqr_pdays = q3_pdays - q1_pdays
lb_pdays = q1_pdays - (1.5 * iqr_pdays)
ub_pdays = q3_pdays + (1.5 * iqr_pdays)

print(f"Any values < {lb_pdays} : {df.loc[df['pdays'] < lb_pdays, 'pdays'].unique()}, len:{len(df.loc[df['pdays'] < lb_pdays, 'pdays'].unique())}")
print(f"Any values > {ub_pdays} : {df.loc[df['pdays'] > ub_pdays, 'pdays'].unique()}, len:{len(df.loc[df['pdays'] > ub_pdays, 'pdays'].unique())}")


# In[23]:


df.boxplot(column="pdays")


# ### Outliers in `previous`

# In[24]:


q1_previous = df["previous"].quantile(0.25)
q3_previous = df["previous"].quantile(0.75)

iqr_previous = q3_previous - q1_previous
lb_previous = q1_previous - (1.5 * iqr_previous)
ub_previous = q3_previous + (1.5 * iqr_previous)

print(f"Any values < {lb_previous} : {df.loc[df['previous'] < lb_previous, 'previous'].unique()}, len: {len(df.loc[df['previous'] < lb_previous, 'previous'].unique())}")
print(f"Any values > {ub_previous} : {df.loc[df['previous'] > ub_previous, 'previous'].unique()}, len: {len(df.loc[df['previous'] > ub_previous, 'previous'].unique())}")


# In[25]:


df.boxplot(column="previous")


# # Observations
# 
# * The `age` field has an upper bound of 70 years.
# 
# * Negative amount are present in `balance` field which is quite suspicious as it denotes the annual income of an individual in euros.
# 
# * The main reason of `balance` field having huge number of outliers is because of the fact that different people as per job and education, earns money on different scale.
# 
# * The `previous` and `pdays` have IQR = 0 which suggest that there is almost zero-variance in the data in `previous` and `pdays`
# 

# # Standarization

# In[26]:


df.loc[df["balance"] < 0, "balance"] = 0
X = df.drop('y', axis=1).values
y = df['y'].values

print(f"Shape of X : {X.shape}")
print(f"Shape of Target data : {y.shape}")


# In[27]:


std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
print(f"Shape of X data : {X_std.shape}")


# # Split the Data into Train and Validation Sets

# In[36]:


from sklearn.model_selection import train_test_split
Xtrain, Xval, Ytrain, Yval = train_test_split(X_std, y, random_state=0, stratify=y)


# # Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)
logreg.fit(Xtrain, Ytrain)

print(f"Training score for Logistic Regression : {logreg.score(Xtrain, Ytrain) * 100.}")
print(f"Validation score for Logistic Regression : {logreg.score(Xval, Yval) * 100.}")


# # Decision Tree

# In[38]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(Xtrain, Ytrain)

print(f"Training Score for Decision Tree : {tree.score(Xtrain, Ytrain) * 100.}")
print(f"Validation Score for Decision Tree : {tree.score(Xval, Yval) * 100.}")


# # Random Forest

# In[39]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(Xtrain, Ytrain)

print(f"Training Score for Random Forest : {rf.score(Xtrain, Ytrain) * 100.}")
print(f"Validation Score for Random Forest : {rf.score(Xval, Yval) * 100.}")


# # XGBoost

# In[40]:


from xgboost import XGBClassifier

xg = XGBClassifier(random_state=0)

xg.fit(Xtrain, Ytrain)

print(f"Training Score for XGBoost : {xg.score(Xtrain, Ytrain) * 100.}")
print(f"Validation Score for XGBoost : {xg.score(Xval, Yval) * 100}")


# # Neural Network

# In[41]:


import torch
import torch.nn as nn


# In[42]:


tensorXtrain = torch.from_numpy(Xtrain).float()
tensorXval = torch.from_numpy(Xval).float()

tensorYtrain = torch.from_numpy(Ytrain).float()
tensorYval = torch.from_numpy(Yval).float()


# In[43]:


# Create PyTorch Dataset
train_dataset = torch.utils.data.TensorDataset(tensorXtrain, tensorYtrain)
val_dataset = torch.utils.data.TensorDataset(tensorXval, tensorYval)


# In[44]:


class ANN(nn.Module):
    
    def __init__(self, n_features, n_classes=1):
        
        super(ANN, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(n_features, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, n_classes))
        
    def forward(self, X):
        out = self.fc(X)
        return out


# In[45]:


# Build the model
n_features = X.shape[1]
n_classes = 1
model = ANN(n_features, n_classes)


# In[46]:


# Build the model losses and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())


# In[47]:


# Create the train and val iterator
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)


# In[48]:


def batch_gd(model, criterion, optimizer, train_iter, val_iter, epochs=20):
    
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    
    train_accs = np.zeros(epochs)
    val_accs = np.zeros(epochs)
    
    for epoch in range(epochs):
        
        train_loss = []
        val_loss = []
        
        train_acc = []
        val_acc = []
        
        for inputs, targets in train_iter:
            
            # Shape the target for BCEWithLogitsLoss 
            targets = targets.view(-1, 1)
            
            # Zero the optimizer-gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and Optimization
            loss.backward()
            optimizer.step()
            
            # Calculate the acc
            acc = np.mean(targets.numpy() == (outputs.detach().numpy() > 0))
            
            # Track the loss and acc
            train_loss.append(loss.item())
            train_acc.append(acc)
            
        
        for inputs, targets in val_iter:
            
            # Shape the target for BCEWithLogitsLoss
            targets = targets.view(-1, 1)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate the acc
            acc = np.mean(targets.numpy() == (outputs.detach().numpy() > 0))
            
            # Track the loss and acc
            val_loss.append(loss.item())
            val_acc.append(acc)
            
        
        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        
        train_acc = np.mean(train_acc)
        val_acc = np.mean(val_acc)
        
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss
        
        train_accs[epoch] = train_acc
        val_accs[epoch] = val_acc
        
        print(f"Epoch : {epoch+1}/{epochs} | Train Losses : {train_loss} | Train Acc : {train_acc} | Val Losses : {val_loss} | Val Acc : {val_acc}")
    
    
    return train_losses, val_losses, train_accs, val_accs
            


# In[49]:


train_losses, val_losses, train_accs, val_accs = batch_gd(model, criterion, optimizer, train_iter, val_iter, epochs=200)


# In[50]:


# Plot the epochs vs losses
plt.title("Epochs vs Losses")
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
plt.show()


# In[51]:


# Plot the epochs vs losses
plt.title("Epochs vs Losses")
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[58]:


def get_accuracy(model, loader):
    
    acc = []
    
    for inputs, targets in loader:
        
        targets = targets.view(-1, 1)
        
        outputs = model(inputs)
        
        predictions = (outputs.detach().numpy() > 0)
        
        acc.append(np.mean(targets.detach().numpy() == predictions))
        
    return np.mean(acc) * 100


# In[59]:


print(f"Training Accuracy of ANN : {get_accuracy(model, train_iter)}")
print(f"Validation Accuracy of ANN : {get_accuracy(model, val_iter)}")


# # Accuracy Comparison of ANN with Other models
# 
# | MODEL               | TRAIN ACC         | VAL ACC           |
# |---------------------|-------------------|-------------------|
# | Logistic Regression | 89.93157956824348 | 89.76377952755905 |
# | Decision Tree       | 100.0             | 87.55197735114571 |
# | Random Forest       | 100.0             | 90.47155622401132 |
# | XGBoost             | 95.67063819747553 | 90.97584712023357 |
# | ANN                 | 91.35806359791803 | 90.25519482857966 |
# 
# 
# 
# * The above table shows that XGBoost is the perfect for our example.
# * The accuracy of prediction of Logistic Regression Model is quite similar to that of Neural Networks
# * Both Decision Tree and Random Forest shows signs of overfitting.

# In[62]:


dump(std_scaler,"./MODEL_DATA/std.bin",compress=True)
dump(xg,"./MODEL_DATA/model.dat")


# In[ ]:





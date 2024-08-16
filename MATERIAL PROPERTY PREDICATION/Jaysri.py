#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required library. 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from xgboost import plot_importance


# In[2]:


#Creating data base from the excel data
df = pd.read_excel(r'C:\Users\srira\OneDrive\Documents\ML_project_data\Project data\UTS.xlsx', sheet_name= 'WRM_DATA -F')
df


# In[3]:


df.groupby(['GRADE','PSN','INPUTSIZE']).count()


# In[4]:


# Display the data types of all columns
max_len = max(len(col) for col in df.columns)  # Find the maximum length of column names

for column in df.columns:
    print(f"{column.ljust(max_len)}: {df[column].dtype}")


# In[5]:


# converting all the columns with data types int to float
for column in df.columns:
    if df[column].dtype == 'int64':
        df[column] = df[column].astype(float)

# Display the data types of all columns
max_len = max(len(col) for col in df.columns)  # Find the maximum length of column names

for column in df.columns:
    print(f"{column.ljust(max_len)}: {df[column].dtype}")


# In[6]:


#Dropping of nan values
df= df.dropna()


# In[7]:


# the 'INPUTSIZE' column will have '160.0 MM' instead of '160MM_160MM'
df['INPUTSIZE'] = df['INPUTSIZE'].replace({'160MM_160MM': '160.0 MM'})
from sklearn.preprocessing import LabelEncoder

# Apply label encoding to INPUTSIZE column
label_encoder = LabelEncoder()
df['INPUTSIZE_encoded'] = label_encoder.fit_transform(df['INPUTSIZE'])

# Use pandas get_dummies function for one-hot encoding
df = pd.get_dummies(df, columns=['Hood Cover Open/close', 'Blower On/off', 'ROUTE'], prefix=['Hood', 'Blower', 'Route'])


# In[8]:


# Understanding the process stability of mill parameters. 

# Step 0: Filter the dataframe for the specific size
specific_size = 13 # Replace with your desired size
filtered_df = df[df['SIZE'] == specific_size]

# Step 1: Extract the subset of columns
selected_columns = ['Act #1 Temp', 'Kocks Entry Temp','UTS IN N/MM2','UTS IN kgf/MM2']

subset_df = filtered_df[selected_columns]

# Step 2: Prepare the data
# Assuming 'INSPECTION DATE' is the datetime index
subset_df.set_index(filtered_df['INSPECTION DATE'], inplace=True)

# Step 3: Create a control chart for each column
for column in subset_df.columns:
    mean = subset_df[column].mean()
    std_dev = subset_df[column].std()
    upper_control_limit = mean + 3 * std_dev
    lower_control_limit = mean - 3 * std_dev
    
    plt.figure(figsize=(10, 6))
    plt.plot(subset_df.index, subset_df[column], marker='o', linestyle='-', label=column)
    plt.axhline(upper_control_limit, color='r', linestyle='--', label='Upper Control Limit')
    plt.axhline(lower_control_limit, color='g', linestyle='--', label='Lower Control Limit')
    plt.axhline(mean, color='b', linestyle='-', label='Mean')
    plt.xlabel('Inspection Date')
    plt.ylabel(column)
    plt.title(f'Control Chart for {specific_size} mm product {column}')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of ylabel
    plt.grid(True)
    plt.show()


# In[9]:


df['Air gas ratio'].unique()


# In[10]:


df['Air gas ratio']= df['Air gas ratio'].replace({'0.6,0.7,0.8':0.7})


# In[11]:


df['Air gas ratio'].unique()
df['Air gas ratio'] = df['Air gas ratio'].astype(float)
df['Air gas ratio'].dtype


# In[12]:


# converting all the columns with data types int to float
for column in df.columns:
    if df[column].dtype == 'int32':
        df[column] = df[column].astype(float)

# Display the data types of all columns
max_len = max(len(col) for col in df.columns)  # Find the maximum length of column names

for column in df.columns:
    print(f"{column.ljust(max_len)}: {df[column].dtype}")


# In[13]:


df.info()


# In[14]:


# converting all the columns with data types int to float
for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(float)

# Display the data types of all columns
max_len = max(len(col) for col in df.columns)  # Find the maximum length of column names

for column in df.columns:
    print(f"{column.ljust(max_len)}: {df[column].dtype}")


# In[15]:


df.columns


# In[16]:


# Filter float columns
float_columns = df.select_dtypes(include='float64').columns

# Calculate correlation matrix
corr_matrix = df[float_columns].corr()

# Round the correlation matrix values to 2 decimal places
rounded_corr_matrix = corr_matrix.round(2)

# Create a heatmap with rounded correlation values
plt.figure(figsize=(40, 30))
sns.heatmap(rounded_corr_matrix, cmap='coolwarm', annot=True)
plt.title("Correlation Heatmap (Float Columns)")
plt.show()


# In[17]:


varaible_columns = [ 'PHZ', 'HZ', 'SZ', 'Air gas ratio', 'Act #1 Temp',
       'Kocks Entry Temp', 'Ntm inlet  temp Actual',
       'Ntm interstand pressure  Act', 'Ntm oulet temperature', 'LHD TEMP',
       ' First conveyor Speed m/s', 'Last conveyor speed m/s', 'C %', 'Si %',
       'Mn %', 'P %', 'S %', 'Al %', 'Cr %', 'Mo %', 'Ni %', 'B PPM', 'Ti %',
       'Ca PPM', 'Pb %', 'V %', 'N PPM', 'O PPM', 'H PPM', 'Nb %', 'Sn %',
       'Cu %', 'Zr %', 'Te %', 'Bi %', 'Sb %', 'As %', 'W %', 'Co %',
       'Al/N Ratio', 'Mn/S Ratio', 'Mn/Si Ratio', 'CE_Long %', 'PCM %',
       'Ni+Cr %', 'Cr+Mo %', 'Cr+Ni+Mo %', 'Cu+Ni %', 'Cu+Ni+Mo %',
       'Ni/Cu Ratio', 'Cu+Ni+Cr+Mo+V %', 'Cu+10Sn %', 'Cu+8Sn %', 'Sn+As+Sb %',
       'Cu+Cr+Ni+Mo %', 'Cu+Cr+Ni %', 'Ti/N Ratio', 'Nb+V+Ti %', 'Nb+V+Ti+B %',
       'Nb+V+Ti+Cu+Mo %', 'Nb+V+Ti+Mo+Cr %', 'Ti+Al+Zr %','Route_NVD', 'Route_VD' ]
variable=df[varaible_columns]
output=df['UTS IN N/MM2']


# Identifying the features for the model

# In[18]:


# Add constant term for the intercept
variable = sm.add_constant(variable)

# Initialize the linear regression model
model = sm.OLS(output, variable)

# Fit the model
result = model.fit()

# Get initial summary
print(result.summary())


# In[ ]:





# In[19]:


#backward elimination of the less important features. 
while True:
    max_p_value = result.pvalues.idxmax()
    max_p_value_value = result.pvalues[max_p_value]
    
    if max_p_value == 'const' or max_p_value_value <= 0.05:
        break
    
    variable.drop(columns=[max_p_value], inplace=True)
    model = sm.OLS(output, variable)
    result = model.fit()
    # Get final summary
print(result.summary())


# In[20]:


#selected features for model creation. 
variable.columns


# In[21]:


# Filter float columns
float_columns = variable.select_dtypes(include='float64').columns

# Calculate correlation matrix
corr_matrix = variable[float_columns].corr()

# Round the correlation matrix values to 2 decimal places
rounded_corr_matrix = corr_matrix.round(2)

# Create a heatmap with rounded correlation values
plt.figure(figsize= (25, 15) )
sns.heatmap(rounded_corr_matrix, cmap='coolwarm', annot=True)
plt.title("Correlation Heatmap (Float Columns)")
plt.show()


# In[ ]:





# In[23]:


#preparing data for model 
x_columns = ['PHZ', 'HZ', 'SZ', 'Air gas ratio', 'Act #1 Temp',
       'Kocks Entry Temp', 'Ntm inlet  temp Actual',
       'Ntm interstand pressure  Act', 'Ntm oulet temperature', 'LHD TEMP',
       ' First conveyor Speed m/s', 'Last conveyor speed m/s', 'C %', 'Si %',
       'Mn %', 'P %', 'S %', 'Al %', 'Cr %', 'Mo %', 'Ni %', 'B PPM', 'Ti %',
       'Ca PPM', 'Pb %', 'V %', 'N PPM', 'O PPM', 'H PPM', 'Nb %', 'Sn %',
       'Cu %', 'Zr %', 'Te %', 'Bi %', 'Sb %', 'As %', 'W %', 'Co %',
       'Al/N Ratio', 'Mn/S Ratio', 'Mn/Si Ratio', 'CE_Long %', 'PCM %',
       'Ni+Cr %', 'Cr+Mo %', 'Cr+Ni+Mo %', 'Cu+Ni %', 'Cu+Ni+Mo %',
       'Ni/Cu Ratio', 'Cu+Ni+Cr+Mo+V %', 'Cu+10Sn %', 'Cu+8Sn %', 'Sn+As+Sb %',
       'Cu+Cr+Ni+Mo %', 'Cu+Cr+Ni %', 'Ti/N Ratio', 'Nb+V+Ti %', 'Nb+V+Ti+B %',
       'Nb+V+Ti+Cu+Mo %', 'Nb+V+Ti+Mo+Cr %', 'Ti+Al+Zr %', 'INPUTSIZE_encoded', 'Hood_CLOSE', 'Hood_OPEN', 'Blower_OFF', 'Blower_ON',
       'Route_NVD', 'Route_VD']
X= df[x_columns]
Y=df['UTS IN N/MM2']


# In[24]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[25]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:





# In[ ]:





# In[35]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





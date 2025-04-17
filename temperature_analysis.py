# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data/GlobalLandTemperaturesByCity.csv')
df.head()

# Data cleaning
df = df.dropna()
df['dt'] = pd.to_datetime(df['dt'])
df['Year'] = df['dt'].dt.year
df['Month'] = df['dt'].dt.month
df['Day'] = df['dt'].dt.day
df = df[df['Country'] == 'India']  # Filter for India
df.head()

# Visualizing temperature over years for India
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Year', y='AverageTemperature', ci=None)
plt.title('Average Temperature in India Over the Years')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.show()

# Predicting future temperatures using Linear Regression
df_temp = df.groupby('Year')['AverageTemperature'].mean().reset_index()
X = df_temp[['Year']]
y = df_temp['AverageTemperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict future temperatures
future_years = pd.DataFrame({'Year': list(range(2023, 2031))})
future_predictions = model.predict(future_years)

# Plot actual vs predicted
plt.figure(figsize=(12,6))
plt.plot(df_temp['Year'], df_temp['AverageTemperature'], label='Actual')
plt.plot(future_years['Year'], future_predictions, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Temperature (India)')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.show()

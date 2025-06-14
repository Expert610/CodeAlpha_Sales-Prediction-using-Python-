import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df =pd.read_csv('Advertising.csv')
df.head()

df.dropna(inplace=True)
df.head()
df.columns

df.drop(columns=["Unnamed: 0"],axis=1,inplace=True)

df.head()

sns.pairplot(df)
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

coefficients = pd.Series(model.coef_, index=X.columns)
print(coefficients)

coefficients.plot(kind='barh', title="Feature Impact on Sales", color='skyblue')
plt.xlabel("Coefficient")
plt.grid(True)
plt.show()

# plt.scatter(df['TV'], df['Sales'])
# plt.xlabel("TV Ad Spend")
# plt.ylabel("Sales")
# plt.title("TV Spend vs Sales")
# plt.grid(True)
# plt.show()


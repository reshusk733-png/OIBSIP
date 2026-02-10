Python 3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import pandas as pd
... import seaborn as sns
... import matplotlib.pyplot as plt
... from sklearn.model_selection import train_test_split
... from sklearn.preprocessing import StandardScaler
... from sklearn.neighbors import KNeighborsClassifier
... from sklearn.metrics import classification_report, accuracy_score
... 
... # 1. Load the Dataset
... # We'll use seaborn's built-in iris loader for convenience
... df = sns.load_dataset('iris')
... 
... # 2. Exploratory Data Analysis (Quick Look)
... print("First 5 rows of the dataset:")
... print(df.head())
... 
... # 3. Data Preprocessing
... # Split data into features (X) and target (y)
... X = df.drop('species', axis=1)
... y = df['species']
... 
... # Split into Training (80%) and Testing (20%) sets
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Scale the features for better model performance
... scaler = StandardScaler()
... X_train = scaler.fit_transform(X_train)
... X_test = scaler.transform(X_test)
... 
... # 4. Model Training (Using K-Nearest Neighbors)
... model = KNeighborsClassifier(n_neighbors=3)
... model.fit(X_train, y_train)
... 
... # 5. Model Evaluation
... predictions = model.predict(X_test)
... accuracy = accuracy_score(y_test, predictions)
... 
... print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
... print("\nClassification Report:")
... print(classification_report(y_test, predictions))
... 
... # 6. Visualization: Pairplot to see clusters
... sns.pairplot(df, hue='species', palette='Dark2')

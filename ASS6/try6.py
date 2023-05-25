# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define function to remove outliers
def RemoveOutliers(df, var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr

    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

# Define function to display outliers
def DisplayOutliers(df, msg):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(msg)
    sns.boxplot(data = df, x = 'sepal.length', ax = axes[0, 0])
    sns.boxplot(data = df, x = 'sepal.width', ax = axes[0, 1])
    sns.boxplot(data = df, x = 'petal.length', ax = axes[1, 0])
    sns.boxplot(data = df, x = 'petal.width', ax = axes[1, 1])
    fig.tight_layout()
    plt.show()

# Define function for each menu option
def display_dataset_info():
    print('Information of dataset:\n', df.info)
    print('Shape of dataset (row x column):', df.shape)
    print('Column names: ', df.columns)
    print('Total elements in dataset:', df.size)
    print('Datatypes of attributes:\n', df.dtypes)
    print('First 5 rows:\n', df.head().T)
    print('Last 5 rows:\n', df.tail().T)
    print('Any 5 rows:\n', df.sample(5).T)

def find_and_remove_outliers(df):
    print('Finding and removing outliers')
    DisplayOutliers(df, 'Before removing outliers')
    df = RemoveOutliers(df, 'sepal.length')
    df = RemoveOutliers(df, 'sepal.width')
    df = RemoveOutliers(df, 'petal.length')
    df = RemoveOutliers(df, 'petal.width')
    DisplayOutliers(df, 'After removing outliers')

def display_correlation_matrix(df):
    #Encoding of output variable
    df['variety'] = df['variety'].astype('category')
    df['variety'] = df['variety'].cat.codes

    # #Find correlation matrix
    print('Finding correlation matrix using heatmap: ')
    sns.heatmap(df.corr(), annot=True)
    plt.show()

def train_and_test_model(df):

    #Split the data into inputs and outputs
    x = df.iloc[:, [0,1,2,3]].values
    y = df.iloc[:, 4].values

    #Training and testing data
    from sklearn.model_selection import train_test_split

    #Assign test data size 20%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    #Normalization of input data
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)

    #Apply Gaussian Naive Bayes Classifier
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    #Display classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    #Display confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print('confusion_matrix\n', cm)
    sns.heatmap(cm, annot=True)
    plt.show()

# Read dataset
df = pd.read_csv('IRIS1.csv')
print('Dataset loaded')

# Define menu options dictionary
menu_options = {
    '1': display_dataset_info,
    '2': find_and_remove_outliers,
    '3': train_and_test_model,
    '4': display_correlation_matrix,
    '5': exit
}

# Menu loop
while True:
    print("Menu:")
    print("1. Display dataset information")
    print("2. Find and remove outliers")
    print("3. Train and test logistic regression model")
    print("4. display correlation matrix")
    print("5. Exit")
    choice = input("Enter your choice (1-5): ")

    if choice in menu_options:
        menu_options[choice](df)
    else:
        print("Invalid choice. Please enter a number between 1 and 5.")
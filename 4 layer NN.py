import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Layer import Layer, calculate_cost, convert_to_binary, accuracy


# ---------------------------- preparing dataset and devide to training set and validation set ----------------------------
categorical_attr = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic',
                    'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']

df = pd.read_csv("data\Dataset.csv")
df.head()

# Converting Categorical values to scaler values
le = LabelEncoder()
df[categorical_attr] = df[categorical_attr].apply(le.fit_transform, axis=0)
# df.head()

# X: Features, y: Classes
X = np.array(df.iloc[:, :-1])
y = np.array(df['Class'])

# fix output classification
for i in range(len(y)):
    if y[i] == 2:  # M
        y[i] = 1
    elif y[i] == 1:  # L
        y[i] = 0
    elif y[i] == 0:  # H
        y[i] = 1

# Deviding Dataset to training and validation set
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=12)
X_train, X_val = X_train.T, X_val.T
y_train, y_val = y_train.reshape(1, 384), y_val.reshape(1, 96)


# ---------------------------------- parameters and hyperparameters ----------------------------------
m = X_train.shape[1]            # number of samples in training set
alpha = 0.1                     # learning rate
numberOfIteration = 30000


hidden_layer1 = Layer(16, 8)
hidden_layer2 = Layer(8, 4)
hidden_layer3 = Layer(4, 2)
output_layer = Layer(2, 1)

# data to plotting Loss and Accuracy graph
trainingSet3_data = []
evaluatingSet3_data = []
categori_label = ["k", "cost", "accuracy"]


def forward(X):
    hidden_layer1.forward_prop(X)
    hidden_layer1.sigmoid_activation()
    hidden_layer2.forward_prop(hidden_layer1.A)
    hidden_layer2.sigmoid_activation()
    hidden_layer3.forward_prop(hidden_layer2.A)
    hidden_layer3.sigmoid_activation()
    output_layer.forward_prop(hidden_layer3.A)
    output_layer.sigmoid_activation()


def backward():
    output_layer.back_prop(m, y_train, 0, 0)
    hidden_layer3.back_prop(m, y_train, output_layer.dz, output_layer.w)
    hidden_layer2.back_prop(m, y_train, hidden_layer3.dz, hidden_layer3.w)
    hidden_layer1.back_prop(m, y_train, hidden_layer2.dz, hidden_layer2.w)
    output_layer.update_wb(alpha)
    hidden_layer3.update_wb(alpha)
    hidden_layer2.update_wb(alpha)
    hidden_layer1.update_wb(alpha)


for k in range(numberOfIteration):
    # forward propagation
    forward(X_train)
    # cost function for training set
    cost = calculate_cost(output_layer.A, y_train)
    # backward propagation
    backward()

    if (k+1) % 1000 == 0 or k == 0:
        trainingSet3_data.append(
            [k+1, cost, accuracy(output_layer.A, y_train)])
        # Evaluating the model
        forward(X_val)
        evaluatingSet3_data.append(
            [k+1, calculate_cost(output_layer.A, y_val), accuracy(output_layer.A, y_val)])

        if k == numberOfIteration-1:
            print("ts - data= ", trainingSet3_data[-1])
            print("es - data= ", evaluatingSet3_data[-1])


df3 = pd.DataFrame(trainingSet3_data, columns=categori_label)
df3.plot(x='k', y='accuracy', marker='.', title="accuracy of training set")
df3.plot(x='k', y='cost', marker='.', title="cost of training set")

df3 = pd.DataFrame(evaluatingSet3_data, columns=categori_label)
df3.plot(x='k', y='accuracy', marker='.', title="accuracy of validation set")
df3.plot(x='k', y='cost', marker='.', title="cost of validation set")

plt.show()

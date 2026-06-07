import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC


# Load cleaned data

df = pd.read_csv("cleaned_flight_data.csv")


# Convert Airline column to numbers

le = LabelEncoder()

df['IATA_Code_Marketing_Airline'] = le.fit_transform(
    df['IATA_Code_Marketing_Airline']
)


# Convert target variable

target_encoder = LabelEncoder()

df['Delay_Category'] = target_encoder.fit_transform(
    df['Delay_Category']
)


# Features

X = df[
[
'Month',
'DayOfWeek',
'IATA_Code_Marketing_Airline',
'CRSDepTime',
'Distance',
'DepDelay',
'WeatherDelay',
'CarrierDelay',
'LateAircraftDelay'
]
]


# Target

y = df['Delay_Category']


# Split dataset

X_train,X_test,y_train,y_test = train_test_split(

X,
y,

test_size=0.2,

random_state=42

)


print("Training Started...\n")


###############################

# Logistic Regression

###############################

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train,y_train)

pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test,pred_lr)

print("Logistic Regression Accuracy:",acc_lr)


###############################

# Random Forest

###############################

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test,pred_rf)

print("Random Forest Accuracy:",acc_rf)


###############################

# SVC

###############################

svc = SVC()

svc.fit(X_train,y_train)

pred_svc = svc.predict(X_test)

acc_svc = accuracy_score(y_test,pred_svc)

print("SVC Accuracy:",acc_svc)


print("\nCompleted")
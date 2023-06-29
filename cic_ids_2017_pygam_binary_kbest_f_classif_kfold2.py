import numpy as np
import pandas as pd
from pygam import LogisticGAM, f, s
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import Normalizer, OrdinalEncoder, StandardScaler

cids = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# Replacing infinite with nan
cids.replace([np.inf, -np.inf], np.nan, inplace=True)
  
# Dropping all the rows with nan values
cids.dropna(inplace=True)
  
categorical_columns = [' Label']
numberic_columns = [' Destination Port', ' Flow Duration', ' Total Fwd Packets',
       ' Total Backward Packets', 'Total Length of Fwd Packets',
       ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
       ' Fwd Packet Length Std', 'Bwd Packet Length Max',
       ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
       ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
       ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
       'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
       ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
       ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
       ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
       ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
       ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
       ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
       ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
       ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
       ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
       ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
       ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
       ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
       'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
       ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
       ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
       ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']

ct = ColumnTransformer(transformers = [#('normalize', Normalizer(norm='l2'), numberic_columns),
                                          ('standard', StandardScaler(), numberic_columns),
                                       ("label", OrdinalEncoder(), categorical_columns)], remainder = 'passthrough')

ct.fit(cids)
cids_transformed = ct.transform(cids)

X = cids_transformed[:, 0:78]
y = cids_transformed[:, 78]

# Define the number of features for feature selection
k_feature = 10

# Create an instance of SelectKBest with the desired scoring function
k_best = SelectKBest(score_func=f_classif, k=k_feature)

# Feature selection
X_train_selected = k_best.fit_transform(X, y)

# Feature selected infomation
feature0 = k_best.get_support(indices=False)
feature1 = k_best.get_support(indices=True)
print('feature0', feature0)
print('feature1', feature1)

# Define the number of folds for cross-validation
num_folds = 5

# Create the k-fold cross-validation object
kfold = KFold(n_splits=num_folds)

# create lam parameter for PyGAM classifier
lamda = 10.0
lam = np.empty(k_feature)
lam.fill(lamda)

clf = LogisticGAM(lam=lam)
# print('self.coef_[0]', clf.coef_[0])
# print('self.coef_', clf.coef_)

accuracies = []

# Perform feature selection and cross-validation
for train_index, val_index in kfold.split(X_train_selected):
    X_train, X_val = X_train_selected[train_index], X_train_selected[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train and evaluate your model
    clf.fit(X_train, y_train)

    # score = clf.score(X_val_selected, y_val)
    y_pred = clf.predict(X_val)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)

    accuracies.append(accuracy)

print("Accuracies", accuracies)    
average_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy: {average_accuracy:.2f}")
# # Perform k-fold cross-validation
# scores = cross_val_score(clf, X, Y, cv=kfold)

# # Print the accuracy for each fold
# for fold, score in enumerate(scores):
#     print(f"Fold {fold + 1}: Accuracy = {score:.2f}")

# Calculate and print the average accuracy across all folds
# average_accuracy = scores.mean()
# print(f"\nAverage Accuracy: {average_accuracy:.2f}")
# print(f"\nAverage Accuracy: {average_accuracy:.2f}")

import numpy as np
import pandas as pd
from pygam import LogisticGAM, f, s
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, OrdinalEncoder

threshold = 0.2
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

ct = ColumnTransformer(transformers = [('normalize', Normalizer(norm='l2'), numberic_columns),
                                       ("label", OrdinalEncoder(), categorical_columns)], remainder = 'passthrough')

ct.fit(cids)
cids_transformed = ct.transform(cids)

X = cids_transformed[:, 0:78]
Y = cids_transformed[:, 78]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=37)

# Train a Gaussian Mixture classifier on the training set
clf = GaussianMixture(
    n_components=2, covariance_type="tied", max_iter=100, random_state=0, init_params='k-means++'
    , reg_covar=1e-6
)

clf.fit(X_train)

# Train a GAM classifier on the training set
gam = LogisticGAM()
gam.fit(X_train, y_train.ravel())

# Use the trained classifier to predict the classes of the test set
y_pred = clf.predict(X_test)
y_proba = np.array(clf.predict_proba(X_test))

y_pred_merge = np.array(y_pred, copy=True)
X_test_new = []

y_idx_filterd = []
i = 0
for ye in y_proba:
    if (ye[0] > threshold and ye[0] < 1 - threshold):
        y_idx_filterd.append(i)
        X_test_new.append(X_test[i])
    i += 1

y_gam_pred = gam.predict(X_test_new)

j = 0
for ye in y_gam_pred:
    idx = y_idx_filterd[j]
    y_pred_merge[idx] = ye
    j += 1
 
# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Evaluate the accuracy of the combined classifier
accuracy_merge = accuracy_score(y_test, y_pred_merge)
print("Accuracy Merge Model:", accuracy_merge)

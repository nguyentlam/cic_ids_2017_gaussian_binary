from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from pygam import LogisticGAM, s, f
import pandas as pd

threshold = 0.2
cids = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# Replacing infinite with nan
cids.replace([np.inf, -np.inf], np.nan, inplace=True)
  
# Dropping all the rows with nan values
cids.dropna(inplace=True)

print(cids.head(10))
print(cids.describe())
print(cids.shape)
print(cids.columns)
print(type(cids.columns))    
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

print(categorical_columns)
print(numberic_columns)
print('len(numberic_columns)', len(numberic_columns))

ct = ColumnTransformer(transformers = [('normalize', Normalizer(norm='l2'), numberic_columns),
                                       ("label", OrdinalEncoder(), categorical_columns)], remainder = 'passthrough')

ct.fit(cids)
cids_transformed = ct.transform(cids)
print('==============', ct.transformers_)
#print('==============', ct.transformers_[0][1].categories_)
print(cids_transformed[0:3])
print(len(cids_transformed))
print(len(cids_transformed[0]))
X = cids_transformed[:, 0:78]
Y = cids_transformed[:, 78]

print('X[0]', X[0])
print('len(X[0])', len(X[0]))
print('Y[0]', Y[0])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=37)
print('X_train[0:3]', X_train[0:3])
print('X_test[0:3]', X_test[0:3])
print('y_train[0:3]', y_train[0:3])
print('y_test[0:3]', y_test[0:3])
# Train a logistic regression classifier on the training set
# clf = LogisticRegression(penalty=None, C=1e-6, solver='saga', multi_class='ovr', max_iter = 100)
# clf = DecisionTreeClassifier()
clf = MLPClassifier(
    hidden_layer_sizes=(80, 40, 80),
    solver="adam",
    activation='logistic',
    alpha=1e-5,
    max_iter=200,
    learning_rate='constant',
    learning_rate_init=0.2,
    random_state=1,
    verbose=True,
)


# clf = GaussianMixture(
#     n_components=2, covariance_type="tied", max_iter=100, random_state=0, init_params='k-means++'
#     , reg_covar=1e-6
# )

# clf = KMeans(
#     n_clusters=2
# )

# clf = LogisticGAM(
#     s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) 
#     + s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17) + s(18) + s(19) 
#     + s(20) + s(21) + s(22) + s(23) + s(24) + s(25) + s(26) + s(27) + s(28) + s(29)
#     + s(30) + s(31) + s(32) + s(33) + s(34) + s(35) + s(36) + s(37) + s(38) + s(39)
#     + s(40) + s(41) + s(42) + s(43) + s(44) + s(45) + s(46) + s(47) + s(48) + s(49)
#     + s(50) + s(51) + s(52) + s(53) + s(54) + s(55) + s(56) + s(57) + s(58) + s(59)
#     + s(60) + s(61) + s(62) + s(63) + s(64) + s(65) + s(66) + s(67) + s(68) + s(69)
#     + s(70) + s(71) + s(72) + s(73) + s(74) + s(75) + s(76) + s(77)
# )

#clf = LogisticGAM()

clf.fit(X_train, y_train)

# # Use the trained classifier to predict the classes of the test set
y_pred = clf.predict(X_test)
y_val = clf.predict(X_train)

y_proba = np.array(clf.predict_proba(X_test))

y_filterd = []
for y in y_proba:
    if (y[0] > threshold and y[0] < 1 - threshold):
        y_filterd.append(y)
# # Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
validate = accuracy_score(y_train, y_val)
print("Validate:", validate)

print("len(y_filterd)", len(y_filterd))
print("y_filterd", y_filterd[0:10])
# print("AIC:", clf.aic(X_test))
# print("BIC:", clf.bic(X_test))
# print("means_", clf.means_)
# print("weights_", clf.weights_)
# print("covariances_", clf.covariances_)
# print("precisions_", clf.precisions_)
# print("precisions_cholesky_", clf.precisions_cholesky_)
# print("converged_", clf.converged_)
# print("n_iter_", clf.n_iter_)
# print("n_features_in_", clf.n_features_in_)
# print("lower_bound_", clf.lower_bound_)
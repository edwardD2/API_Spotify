# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0]) - 1):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            try:
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            except ZeroDivisionError:
                row[i] = row[i] - minmax[i][0]


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


# locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# extract numerical data, implementing one-hot encoding for track genre
names = ['tteokbokki', 'b', 'never forget', 'whatever simulation', 'untitled', 'i feel like', 'not edgy', 'doggo']
tracks_all = pd.DataFrame()
artists = pd.read_csv('csv files/artists.csv', index_col=0)
top_genres = pd.read_csv('csv files/genres.csv', index_col=0)
df = pd.read_csv('csv files/tracks.csv', index_col=0)
genres = []
ohe = pd.DataFrame(columns=top_genres.all_top_genres.values)
for artist in df.track_artists:
    try:
        genres = artists[artists['artist_name'] == artist]['genres'].values[0]
        genres = genres.replace('[', '').replace(']', '').replace('\'', '').replace(', ', ',').split(',')
    except IndexError:
        genres = []
    a = {}
    for genre in top_genres.all_top_genres.values:
        if genre in genres:
            a[genre] = 1
        else:
            a[genre] = 0
    ohe = ohe.append(a, ignore_index=True)
df = ohe.join(df)[top_genres.all_top_genres.values.tolist() +
                  ['energy', 'loudness', 'tempo', 'valence', 'danceability', 'playlist']]
tracks_all = tracks_all.append(df, ignore_index=True)

# drop last track if total number of tracks not even
if len(tracks_all) % 2 != 0:
    tracks_all = tracks_all.drop([len(tracks_all) - 1])


# sample 50% observations as training data
random.seed(55)
train = random.sample(range(0, len(tracks_all)), len(tracks_all) // 2)
ttrain = tracks_all.index.isin(train)

tracks_train = tracks_all[ttrain]

# rest 50% as test data
tracks_test = tracks_all[~ttrain]

# create response vectors and design matrices for training and test set
y_train = tracks_train['playlist'].tolist()
x_train = tracks_train.drop('playlist', axis=1).values.tolist()

y_test = tracks_test['playlist'].tolist()
x_test = tracks_test.drop('playlist', axis=1).values.tolist()

tracks_train_values = tracks_train.values.tolist()

# normalize datasets
normalize_dataset(tracks_train_values, dataset_minmax(tracks_train_values))
normalize_dataset(x_train, dataset_minmax(x_train))
normalize_dataset(x_test, dataset_minmax(x_test))


# perform k-fold cross validation to select best number of neighbors
def do_chunk(k, folds, x_data, y_data, num_neighbors):
    error_folds = pd.DataFrame(columns=['fold', 'train_error', 'val_error', 'neighbors'])
    for chunk_id in range(k):
        t = [fold != chunk_id for fold in folds]
        trn_indices = [i for i, x in enumerate(t) if x]
        val_indices = [i for i, x in enumerate(t) if not x]
        xtr = [x_data[i] for i in trn_indices]
        xvl = [x_data[i] for i in val_indices]
        ytr = [y_data[i] for i in trn_indices]
        yvl = [y_data[i] for i in val_indices]
        pred_labels_tr = []
        pred_labels_vl = []
        for row in xtr:
            pred_labels_tr.append(predict_classification(xtr, row, num_neighbors))
        for row in xvl:
            pred_labels_vl.append(predict_classification(xtr, row, num_neighbors))
        matrix_tr = confusion_matrix(ytr, pred_labels_tr, labels=names)
        matrix_vl = confusion_matrix(yvl, pred_labels_vl, labels=names)

        trn_err = 1 - np.trace(matrix_tr) / np.sum(matrix_tr)
        val_err = 1 - np.trace(matrix_vl) / np.sum(matrix_vl)

        error_folds = error_folds.append({'fold': chunk_id, 'train_error': trn_err,
                                          'val_error': val_err, 'neighbors': num_neighbors}, ignore_index=True)
        error_folds['fold'] = error_folds['fold'].astype(int)
        error_folds['neighbors'] = error_folds['neighbors'].astype(int)

    return error_folds


random.seed(32)
k = 3
folds = random.sample(list(pd.cut(range(0, len(x_train)), bins=k, labels=False)), len(x_train))
all_neighbors = 50

all_error_folds = pd.DataFrame(columns=['fold', 'train_error', 'val_error', 'neighbors'])
for num_neighbor in range(1, all_neighbors + 1):
    all_error_folds = all_error_folds.append(do_chunk(k=k, folds=folds,
                                                      x_data=tracks_train_values, y_data=y_train,
                                                      num_neighbors=num_neighbor), ignore_index=True)

best_neighbors = all_error_folds.groupby(by='neighbors').mean()['val_error'].idxmin()

# predictions on training set
pred_labels = []
for row in x_train:
    pred_labels.append(predict_classification(tracks_train_values, row, best_neighbors))

matrix = confusion_matrix(y_train, pred_labels, labels=names)

# predictions on test set
pred_labels_test = []
for row in x_test:
    pred_labels_test.append(predict_classification(tracks_train_values, row, best_neighbors))

matrix_test = confusion_matrix(y_test, pred_labels, labels=names)

# visualize confusion matrix
titles_options = [
    ('Confusion matrix, without normalization', None),
    ('Normalized confusion matrix', 'true'),
]
for title, normalize in titles_options:
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred_labels_test,
        labels=names,
        xticks_rotation='vertical',
        normalize=normalize,
        cmap=plt.cm.Blues,
        ax=ax)
    disp.ax_.set_title(title)
plt.show()

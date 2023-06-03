import numpy as np
import pandas as pd


def _create_windowed_data_and_labels(data, labels, window_size=20, shift=1):
    num_features = data.shape[1]
    num_batches = data.shape[0]-window_size + 1

    output = np.repeat(np.nan, num_batches*num_features*window_size).reshape(
        (data.shape[0]-window_size + 1, window_size, num_features))
    output_labels = np.repeat(np.nan, num_batches)

    # select based on window
    for iter in range(num_batches):
        output[iter, :, :] = data[iter:iter+window_size, :]
        output_labels[iter] = labels[iter+window_size-1]

    return output, output_labels


def preprocess_train_windowed_UL(train_data: pd.DataFrame, window_size=20, uuid='unit_ID', label='RUL', cols_to_drop=['unit_ID', 'cycles', 'RUL']):
    basetable_x = []
    basetable_y = []
    train_UL = []

    for id in set(train_data[uuid]):
        temp = train_data.loc[train_data[uuid] == id]

        # shift by one timeunit, predict the rul of the next day
        labels = temp[label].shift(-1)
        labels = labels.loc[~labels.isna()]
        # to make sure the nans are removed from indep var
        temp = temp.loc[labels.index]
        labels = labels.values
        # End shift

        temp = temp.drop(cols_to_drop, inplace=False, axis=1).values

        temp, labels = _create_windowed_data_and_labels(
            temp, labels, window_size=window_size)
        UL = temp[:, -1, -1].reshape((len(temp[:, :, -1]), 1))
        temp = temp[:, :, :-1]

        basetable_x.append(temp)
        basetable_y.append(labels)
        train_UL.append(UL)

    basetable_x = np.vstack(basetable_x)
    basetable_y = np.hstack(basetable_y)
    train_UL = np.vstack(train_UL)

    return basetable_x, basetable_y.reshape((basetable_y.shape[0], 1)), train_UL


def preprocess_test_windowed_UL(test_data, window_size=20, uuid='unit_ID', label='RUL', cols_to_drop=['unit_ID', 'cycles', 'RUL']):
    basetable_x_test = []
    basetable_y_test = []
    basetable_UL_test = []

    for id in set(test_data[uuid]):
        temp = test_data.loc[test_data[uuid] == id]

        # shift by one timeunit, predict the rul of the next day
        labels = temp[label].shift(-1)
        labels = labels.loc[~labels.isna()]
        temp = temp.loc[labels.index]
        labels = labels.values
        # End shift

        temp = temp.drop(cols_to_drop, inplace=False, axis=1).values

        temp, labels = _create_windowed_data_and_labels(
            temp, labels, window_size=window_size)
        UL = temp[:, -1, -1].reshape((len(temp[:, :, -1]), 1))
        temp = temp[:, :, :-1]
        basetable_UL_test.append(UL)
        basetable_x_test.append(temp)
        basetable_y_test.append(labels)

    return basetable_x_test, basetable_y_test, basetable_UL_test

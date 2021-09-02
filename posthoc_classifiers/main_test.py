import pandas as pd
import numpy as np
from utils.classifier_training import parallel_train
from sklearn.model_selection import train_test_split

eps_list = [0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100]
train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

shuffle_cols=False

results_type='dpcopula'
fnames = [
    'logisticregression', 'randomforest', 'gaussiannb', 'bernoullinb',
    'svmrbf', 'gbm', 'extratrees', 'lda', 'passiveagressive', 'adaboost',
    'bagging']
real_label=1
fake_label=0

for name in fnames:
    auc_array =[]
    classifier = name
    for item in train_list:
        dataset_auc_max = []
        for epsilon in eps_list:
            real_data_path = f'../datasets/lifesci.csv'
            fake_data_path = f'../synth_results/{results_type}/datasets_eps_{epsilon}/data_{epsilon}_{item}.csv'


            real_data = pd.read_csv(real_data_path, header=None)
            real_data = real_data[1:] # Remove header from real data
            if shuffle_cols:
                real_data.apply(lambda x: x.sample(frac=1).values)

            fake_data = pd.read_csv(fake_data_path, header=None)

            if results_type == 'datasynthesizer':
                fake_data = fake_data[1:] # Remove header for DataSynthesizer
                fake_data.columns = pd.RangeIndex(fake_data.columns.size)

            if results_type == 'dpwgan':
                fake_data = fake_data.iloc[1:, :] # Remove row index
                fake_data = fake_data.iloc[: , 1:] # Remove col index
                fake_data.columns = pd.RangeIndex(fake_data.columns.size)
                fake_data[10][fake_data[10] < 0.5] = 0 # Threshold attribute
                fake_data[10][fake_data[10] >= 0.5] = 1

            if results_type == 'dp-copula':
                fake_data.columns = pd.RangeIndex(fake_data.columns.size)

            if results_type == 'pategan':
                fake_data.columns = pd.RangeIndex(fake_data.columns.size)
                fake_data.columns = pd.RangeIndex(fake_data.columns.size)
                fake_data[10][fake_data[10] < 0.5] = 0 # Threshold attribute
                fake_data[10][fake_data[10] >= 0.5] = 1


            real_vector = [real_label for i in range(real_data.shape[0])]
            fake_vector = [fake_label for i in range(fake_data.shape[0])]

            real_data[11] = real_vector #attach class label
            fake_data[11] = fake_vector #attach class label

            if shuffle_cols:
                fake_data.apply(lambda x: x.sample(frac=1).values)

            frames = [real_data, fake_data]

            concat = pd.concat(frames, sort=False)
            shuffle = concat.sample(frac=1).reset_index(drop=True) #shuffle dataset
            y = shuffle.iloc[:, [-1]] #detach shuffled labels
            X = shuffle.drop([11], axis=1) #drop label col

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            y_train = np.ravel(y_train)
            y_test = np.ravel(y_test)

            max_auc = parallel_train(X_train, y_train, X_test, y_test, classifier, 10)
            dataset_auc_max.append(max_auc)
        auc_array.append(dataset_auc_max)

    if shuffle_cols:
        np.savetxt(f'../posthoc_results/{results_type}_shuffle_aucs/{classifier}.csv', auc_array, delimiter=',')
    else:
        np.savetxt(f'../posthoc_results/{results_type}_aucs/{classifier}.csv', auc_array, delimiter=',')

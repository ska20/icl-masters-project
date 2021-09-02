import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'pategan_aucs' # Data to be plotted

'''
Formatting to ensure proper graph title name
'''

if data_dir == 'pategan_aucs':
    title_name = 'PATE-GAN'
elif data_dir == 'dpcopula_aucs':
    title_name = 'DP-Copula'
elif data_dir == 'datasynthesizer_aucs':
    title_name = 'DP-Synthesizer'
elif data_dir == 'dpwgan_aucs':
    title_name = 'DP-WGAN'

'''
Strings for filepath building
'''

fnames = [
    'logisticregression', 'randomforest', 'gaussiannb', 'bernoullinb',
    'svmrbf', 'gbm', 'extratrees', 'lda', 'passiveagressive', 'adaboost',
    'bagging', 'nn']
suffix = 'csv'

'''
List of privacy budgets to iterate over
'''

eps_list = [0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100]

'''
matplotlib formatting for consistent graph style
'''

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']

markers = ['v', '^', '>', '<', '1', '2', '3', '4']

'''
matplotlib formatting for proper legend
'''

name_dict = {
    'logisticregression': 'Logistic Regression',
    'randomforest': 'Random Forest Classifier',
    'gaussiannb': 'Gaussian Naive Bayes',
    'bernoullinb': 'Bernoulli Naive Bayes',
    'svmrbf': 'Support Vector Machine - RBF Kernel',
    'gbm': 'Gradient Boosting Classifier',
    'extratrees': 'Extra Trees Classifier',
    'lda': 'Linear Discriminant Analysis',
    'passiveagressive': 'Passive Aggressive Classifier',
    'adaboost': 'AdaBoost Classifier',
    'bagging': 'Bagging Classifier',
    'nn': 'Neural Network Classifier'
}

plt.style.use('default')
plt.xscale('log') # Graph on a log scale
plt.xlabel('Epsilon')  # Add an x-label to the axes.
plt.ylabel('Score')  # Add a y-label to the axes.
plt.title(f'Average AUC Score by Epsilon - {title_name}')  # Add a title to the axes.
hull = np.zeros(9)
for name in fnames:
    fpath = f'../posthoc_results/{data_dir}/{name}.{suffix}' # Generate filename
    data = pd.read_csv(fpath) # Read data
    data = data.to_numpy()
    avg = np.mean(data, axis=0) # Compute average across all 10 measurements
    hull = np.maximum(hull, avg)  # Find maximum data point for each epsilon
    plt.plot(eps_list, avg, alpha=0.5, color='silver') # Plot all classifiers in grey

plotted = [] # list of graphs that have been plotted in color
j=0

for name in fnames:
    fpath = f'../posthoc_results/{data_dir}/{name}.{suffix}' # Generate filename
    data = pd.read_csv(fpath) # Read data
    data = data.to_numpy()
    avg = np.mean(data, axis=0) # Compute average
    for i in range(len(hull)):
        if avg[i] == hull[i] and name not in plotted: # If a classifier contributes to the hull and has not been plotted in color
            print(f'{name_dict[name]}: {avg}') # Print values to command line
            plt.plot(eps_list, avg, linewidth=3, label=name_dict[name], color = colors[j], marker = markers[j]) # Plot in color
            plotted.append(name) # Append name to plotted list
            j+=1
print(hull) # Print hull values to command line
plt.plot(eps_list, hull, linewidth=3, label='Maximum', color='#d62728', linestyle='dotted', marker='o') # Plot maximum line

plt.ylim(0.5, 1.01) # Set plotting limits
plt.grid()
plt.legend()
plt.savefig(f'{data_dir}.png') # Save figure
plt.show()

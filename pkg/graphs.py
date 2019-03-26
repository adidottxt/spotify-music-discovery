'''
graphs
'''

import matplotlib.pyplot as plt  # pylint: disable=import-error
import matplotlib.ticker as mtick  # pylint: disable=import-error

from .constants import NUM_CLUSTERS

def get_new_results(algorithm, results, labels):
    '''
    get new results
    '''
    new_results = {}
    for i in range(len(results)):
        new_results[labels[i]] = {}
        for k in results[i][algorithm]: results[labels[i]][k] = results[i][algorithm][k]
    return new_results

def get_plottable_results(data):
    '''
    get plottable results
    '''
    y_results, labels = [], []
    for k in data:
        for value in data[k]:
            y_results.append(data[k][value])
            labels.append(k + ' ' + (' Random' if value ==
                                     'random' else ' Uncertainty'))
    return y_results, labels


def accuracy_plot(y_values, labels, title, benchmark, save=False):
    '''
    accuracy plot
    '''

    x_values = [i + 5 for i in range(len(y_values[0]))]

    for i in enumerate(y_values):
        plt.plot(
            x_values,
            y_values[i],
            '--' if i %
            2 == 0 else '-',
            label=labels[i])
    plt.plot(x_values, [benchmark for i in range(len(y_values[0]))],
             '-', label='Benchmark', color='black')
    plt.legend()
    plt.title(title)
    plt.xlabel('# Training Instances')
    plt.ylabel('Accuracy')
    plt.axis([5, 5 + len(y_values[0]), 0, 1.0])
    plt.axes().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if save:
        plt.savefig(
            'plots/' +
            str(NUM_CLUSTERS) +
            '_' +
            title.replace(
                ' ',
                '_') +
            '.png')
    plt.show()


def accuracy_plot_c(y_values, labels, title, benchmark, save=False):
    '''
    accuracy plot clusters
    '''

    x_values = [i + 5 for i in range(len(y_values[0]))]

    for i in enumerate(y_values):
        plt.plot(
            x_values,
            y_values[i],
            '--' if i %
            2 == 0 else '-',
            label=labels[i])
    plt.plot(x_values, [benchmark for i in range(len(y_values[0]))],
             '-', label='Benchmark', color='black')
    plt.legend()
    plt.title(title)
    plt.xlabel('# Training Instances')
    plt.ylabel('Accuracy')
    plt.axis([5, 5 + len(y_values[0]), 0, 1.0])
    plt.axes().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if save:
        plt.savefig(
            'plots/' +
            str(NUM_CLUSTERS) +
            '_' +
            title.replace(
                ' ',
                '_') +
            '.png')
    plt.show()

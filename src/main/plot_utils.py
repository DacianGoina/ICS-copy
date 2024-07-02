import matplotlib
import matplotlib.pyplot as plt

'''
This module contains functions used to create specific plots such as histograms.
'''

# Use this to provide a canvas for plotting
matplotlib.use('TkAgg')

def plot_ts_data(x_values, y_values, x_scatter = None, y_scatter = None):
    '''
    Plot data given by x_values, y_values. Optionally, you can scatter points on the plot
    :param x_values:
    :param y_values:
    :param x_scatter:
    :param y_scatter:
    :return:
    '''
    plt.figure(figsize=(18,4))
    if x_scatter is not None and x_scatter is not None:
        plt.scatter(x_scatter, y_scatter, color = 'red')
    plt.plot(x_values, y_values)
    plt.show()

def plot_frequencies_barplot(x_values, y_values):
    '''
    Create and display a barplot. Usually this is used to plot frequencies values:
    items values on x-axis and frequencies on y-axis
    :param x_values:
    :param y_values:
    :return:
    '''
    plt.figure(figsize=(14,6))
    plt.bar(x_values, y_values)

    # and ticks values and labels
    plt.xticks(x_values, x_values)
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('Freq of anomalous points positions in anomalous cycles')

    plt.show()

def plot_hist(values, bins_no = 35, alpha_val = 0.6):
    '''
    Plot historgram for the given values
    :param values:
    :param bins_no:
    :param alpha_val:
    :return:
    '''
    plt.figure(figsize=(9,4))
    plt.hist(values, bins=bins_no, density=True, alpha=alpha_val, color='g', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution Histogram')
    plt.show()
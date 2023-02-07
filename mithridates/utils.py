import numpy as np


def get_inflection_point(analysis, simplified=False, plot=False):
    results = [(x.config['poison_ratio'], x.last_result['backdoor_accuracy']) for x in
               analysis.trials]
    sorted_data = np.array(sorted(results, key=lambda x: x[0]))
    accuracies = sorted_data[:, 1]
    poison_ratios = 100 * sorted_data[:, 0]

    if simplified:
        threshold = 50
        print(f'Using simplified threshold: of {threshold} for backdoor accuracy.')
        infl = np.argmax(sorted_data[:, 1] > threshold)
        if plot:
            plot_it(poison_ratios, accuracies, infl=infl)
    else:
        from scipy.ndimage import gaussian_filter1d
        print(f'Looking for inflection point with second derivative')
        smooth = gaussian_filter1d(accuracies, 2)

        # compute second derivative
        smooth_d2 = np.gradient(np.gradient(smooth))

        # find inflection points
        infls = np.where(np.diff(np.sign(smooth_d2)))[0]
        infl_clean = list()
        for i, infl in enumerate(infls, 1):
            if 15 < accuracies[infl] < 90: # ignore fluctuation outside of range.
                infl_clean.append(infl)
        if len(infl_clean) > 1:
            print(f'Multiple inflection points found: {infl_clean}.')
        infl = infl_clean[0]
        if plot:
            plot_it(poison_ratios, accuracies, smooth, smooth_d2, infl)

    print(f'Inflection point is at {poison_ratios[infl]:2.3f}% of the training dataset '
          f'and backdoor accuracy: {accuracies[infl]:2.3f}%')
    return poison_ratios, accuracies, infl


def plot_it(poison_ratios, accuracies, smooth=None, smooth_d2=None, infl=None):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(poison_ratios, accuracies, label='Backdoor Accuracy')

    ax1.set_xscale('log')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0.01, 10)

    ax1.set_xlabel('Poison ratio, % of dataset')
    ax1.set_ylabel('Backdoor Accuracy', color='b')
    if infl is not None:
        ax1.axvline(x=poison_ratios[infl], color='k', label=f'Inflection Point')
    if smooth is not None and smooth_d2 is not None:
        ax2 = ax1.twinx()
        ax1.plot(poison_ratios, smooth, label='Smoothed Accuracy', c='orange')
        ax2.plot(poison_ratios, smooth_d2 / np.max(smooth_d2),
                 label='Second Derivative (scaled)', c='green')

        ax2.set_ylim(-1, 1)
        ax2.set_ylabel('Second Derivative', color='g')
        ax2.legend(loc='lower right')
    ax1.legend(loc='upper right')


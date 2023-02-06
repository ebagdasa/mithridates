import numpy as np


def get_inflection_point(analysis, simplified=False):
    if simplified:
        pp = dict()
        for x in analysis.trials:
            if x.is_finished():
                pp[x.config['poisoning_proportion']] = x.last_result['backdoor_error']
        min_error = 50  #  offset
        for pois_prop, error in sorted(pp.items(), key=lambda x: x[0]):
            if error <= min_error:
                return pois_prop
    else:
        from scipy.ndimage import gaussian_filter1d
        results = [(x.last_result['poisoning_proportion'], x.last_result['backdoor_accuracy']) for x in
                   analysis.trials]
        sorted_data = np.array(sorted(results, key=lambda x: x[0]))
        raw = sorted_data[:, 1] / 100

        smooth = gaussian_filter1d(raw, 2)

        # compute second derivative
        smooth_d2 = np.gradient(np.gradient(smooth))

        # find inflection points
        infls = np.where(np.diff(np.sign(smooth_d2)))[0]

        return sorted_data[infls[0]]
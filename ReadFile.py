import numpy as np

def ReadData(data_dir, k):
    with open(data_dir) as f:
        num_example = int(f.readline())
        line = f.readline()
        labels = []
        feature_vectors = []
        while line:
            parse_line = [v.strip() for v in line.split(',')]
            feature = np.asarray([float(i) for i in parse_line[6:211]])
            feature_vectors = np.concatenate((feature_vectors, feature), axis=0)
            labels = np.concatenate((labels, [int(parse_line[212])]), axis=0)
            line = f.readline()

        feature_vectors = np.reshape(feature_vectors, (num_example, 205))
    return {'_images': feature_vectors, '_labels': labels, '_epochs_completed': 0, '_index_in_epoch': 0, '_num_examples': num_example-1,
            'images': feature_vectors, 'labels': labels, 'epochs_completed': 0, 'index_in_epoch': 0}
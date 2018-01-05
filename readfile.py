import numpy as np

def read_data(data_dir):
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

def next_batch(data, batch_size, shuffle=True):
    start = data['_index_in_epoch']
    # Shuffle for the first epoch
    if data['_epochs_completed'] == 0 and start == 0 and shuffle:
        perm0 = np.arange(data['_num_examples'])
        np.random.shuffle(perm0)
        data['_images'] = data['images'][perm0]
        data['_labels'] = data['labels'][perm0]
    # Go to the next epoch
    if start + batch_size > data['_num_examples']:
        # Finished epoch
        data['_epochs_completed'] += 1
        # Get the rest examples in this epoch
        rest_num_examples = data['_num_examples'] - start
        images_rest_part = data['_images'][start:data['_num_examples']]
        labels_rest_part = data['_labels'][start:data['_num_examples']]
        # Shuffle the data
        if shuffle:
            perm = np.arange(data['_num_examples'])
            np.random.shuffle(perm)
            data['_images'] = data['images'][perm]
            data['_labels'] = data['labels'][perm]
        # Start next epoch
        start = 0
        data['_index_in_epoch'] = batch_size - rest_num_examples
        end = data['_index_in_epoch']
        images_new_part = data['_images'][start:end]
        labels_new_part = data['_labels'][start:end]
        return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
        data['_index_in_epoch'] += batch_size
        end = data['_index_in_epoch']
        return data['_images'][start:end], data['_labels'][start:end]

def generate_k_folds(data_dir, k):
    line_list = []
    input_file = data_dir + "total.txt"
    with open(input_file) as f:
        num_line = int(f.readline())
        perm0 = np.arange(num_line)
        np.random.shuffle(perm0)

        line = f.readline()
        while line:
            line_list.append(line)
            line = f.readline()

    for i in range(0, k):
        file_dir = data_dir + "kfolds/" + str(i) + ".txt"
        with open(file_dir, 'w') as file:
            start = int(num_line/k)*i
            end = int(num_line/k)*(i+1)
            #file.write(str(end-start)+'\n')
            for j in range(start, end):
                file.write(line_list[perm0[j]])
            file.closed
    print("Generated dataset!")
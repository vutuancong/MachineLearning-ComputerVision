def load_data(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    feature_data = []
    label_data = []
    for line in lines:
        str_arr = line.split("\t")
        float_arr = map(float, str_arr)
        feature_data.append(float_arr[:-1])
        label_data.append(float_arr[-1])
    return feature_data, label_data

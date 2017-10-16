def get_data_by_label(data, labels, expect_label):
	xs = []
	ys = []
	zs = []
	feature_size = len(data[0])
	for id_, data in enumerate(data):
		if labels[id_] == expect_label:
			if feature_size >= 1:
				xs.append(data[0])
			if feature_size >= 2:
				ys.append(data[1])
			if feature_size >= 3:
				zs.append(data[2])	
	return xs, ys, zs
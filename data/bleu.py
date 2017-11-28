def load_file(fn):
	with open(fn) as f:
    	lines = f.readlines()
    return lines

predictions = load_file('preds.txt')
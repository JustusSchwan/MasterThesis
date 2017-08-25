import cPickle
import pprint
import operator
import csv
import numpy as np

csv_file = open('D:/Master Thesis/Dataset/pos_features.csv', 'rb')
csv_reader = csv.reader(csv_file, delimiter=',')

feature_desc = ()

# Get data size and lines for each individual
for line_num, l in enumerate(csv_reader):
    if line_num == 0:
        feature_desc = np.array(l[:-2], dtype=np.str)
        break

loadfile = open('D:/Master Thesis/Dataset/pos_feature_selection_even_sampling.pkl', 'rb')
feature_selection = cPickle.load(loadfile)

pprint.pprint(sorted([['-'.join(feature_desc[excluded].tolist()), score] for excluded, score in feature_selection],
                     key=operator.itemgetter(1)), width=100)

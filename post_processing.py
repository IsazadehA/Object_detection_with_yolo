# read a json file containing x and y coordinates of the points and plot it 
# using matplotlib
import json
import matplotlib.pyplot as plt
import numpy as np

# read the json file
with open('Results/val_cls_loss_chart_data_small.json') as f:
    data = json.load(f)

# extract x and y coordinates
x = [d['x'] for d in data]
y = [d['y'] for d in data]
# convert to numpy arrays
x = np.array(x)
y = np.array(y)
# convert to float
x = x.astype(np.float)
y = y.astype(np.float)
# remove nan values
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]
# make sure x and y have the same length
y = y[0:len(x)]
y.shape
x.shape

x
y

# plot the points using matplotlib, color blue, linestyle dashed
plt.plot(x, y, color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
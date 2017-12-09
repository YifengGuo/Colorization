import matplotlib.pyplot as plt
import numpy as np
import csv

color_list = []

with open('color.csv') as f:
    reader = csv.reader(f)
    color_list = list(reader)

new_color_list = []


for line in color_list:
    new_line = []
    for value in line:
        value = int(value) / 255
        new_line.append(value)
    new_color_list.append(new_line)
    print(new_line)

x, y = np.random.random((2, 100000))
rgb = np.random.random((10, 3))
print(rgb)
fig, ax = plt.subplots()
ax.scatter(x, y, s=.5, facecolors=new_color_list)
plt.show()
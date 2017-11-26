import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('color.csv')
plt.plot(df, label='Test1')
plt.show()

# import csv
#
# with open('color.csv') as file:
#     lines=csv.reader(file)
#     for line in lines:
#         print(line)
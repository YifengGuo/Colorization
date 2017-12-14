from PIL import Image
import csv
color_list = []
fileIn = open('out_reg.csv')
for line in fileIn.readlines():
    lineArr = line.strip().split(',')
    color_list.append(tuple(map(int, map(float, lineArr))))

for tuple in color_list:
    print tuple
    print type(tuple)

im = Image.new("RGB", (221, 221))  # pixels of image
pix = im.load()
i = 0

for x in range(221):
    for y in range(221):
        pix[x, y] = color_list[i]
        i += 1

im.save("outreg.png", "PNG")
print 'finished'
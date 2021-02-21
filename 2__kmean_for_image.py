import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

img = plt.imread('a.jpg')

height = img.shape[0]
width = img.shape[1]

img = img.reshape(height*width,3)

kmeans = KMeans(n_clusters=4).fit(img)
clusters = kmeans.cluster_centers_
labels = kmeans.labels_

# assign 4 cluster to each pixel
# way 1:
# img2 = numpy.zeros_like(img)
# for i in range(len(img2)):
#     img2[i] = clusters[labels[i]]
# img2 = img2.reshape(height,width,3)

# way 2
img2 = numpy.zeros((height,width,3),dtype = numpy.uint8)
index = 0
for i in range(height):
    for j in range(width):
        img2[i][j] = clusters[labels[index]]
        index += 1

plt.imshow(img2)
plt.show()
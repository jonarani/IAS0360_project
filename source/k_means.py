import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import shutil

# Count unique values in matrix
def count_unique(matrix, position):
    counter = 0
    np_unique = np.unique(matrix)
    if len(np_unique) > (position - 1):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == np_unique[position - 1]:
                    counter += 1
    return counter

# Get k-means image in gray scale
def get_k_means(img, K, attempts):
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret,label,center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    # Convert to gray scale for easier warm object detection (less unique values) 
    return cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

# Analyze image using k-means clustering
def analyze_image(fpath, K, attempts):
    img = cv2.imread(fpath, cv2.COLOR_BGR2RGB)
    gray = get_k_means(img, K, attempts)

    picture_size = len(gray) * len(gray[0])
    warm_object_size = float((count_unique(gray, 3) / picture_size)*100)
    percentage_string = "{:.1f}".format(float((count_unique(gray, 1) / picture_size)*100)) + ",{:.1f}".format(float((count_unique(gray, 2) / picture_size)*100)) + ",{:.1f}".format(warm_object_size) + ",{:.1f}".format(float((count_unique(gray, 4) / picture_size)*100))
    print(str(fpath) + ": " + str(np.unique(gray)) + ": " + percentage_string)

    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(gray)
    plt.title('Clustered image when K = %i' % K), plt.xticks([]), plt.yticks([])
    plt.show()

# Analyze folder of images using k-means clustering
def analyze_folder(dpath, K, attempts):
    counter = 0
    k_file = open("kmeans.txt", "w")
    for fname in os.listdir(dpath):
        fpath = os.path.join(dpath, fname)
        counter += 1
        img = cv2.imread(fpath, cv2.COLOR_BGR2RGB)
        gray = get_k_means(img, K, attempts)

        picture_size = len(gray) * len(gray[0])
        warm_object_size = float((count_unique(gray, 3) / picture_size)*100)
        percentage_string = "{:.1f}".format(float((count_unique(gray, 1) / picture_size)*100)) + ",{:.1f}".format(float((count_unique(gray, 2) / picture_size)*100)) + ",{:.1f}".format(warm_object_size) + ",{:.1f}".format(float((count_unique(gray, 4) / picture_size)*100))
        print(str(fpath) + ": " + str(np.unique(gray)) + ": " + percentage_string)
        k_file.write(str(fpath) + "," + str(np.unique(gray)) + "," + percentage_string + '\n')
    k_file.close()

# Move images to correct folder based on output of k-mean analyze_folder
def image_classification(kfile):
    with open(kfile) as f:
        line_count = len(open(kfile).readlines())
        for i in range(line_count):
            line = f.readline()
            splitted_line = line.split(',')
            warm_object = float(splitted_line[4])
            if (warm_object < 1):
                print(splitted_line[0] + ": " + splitted_line[4])        
            if warm_object > 0.5 and warm_object < 10:
                shutil.copy(splitted_line[0], os.path.join('imgs_human', splitted_line[0].split('\\')[1]))
            else:
                shutil.copy(splitted_line[0], os.path.join('imgs_background', splitted_line[0].split('\\')[1]))
def main():
    #analyze_image(os.path.join("../dataset/images_from_matlab_script", "fig_20210628_1630_3078_0376_10_22.png"), 4,10)
    analyze_folder("../dataset/images_from_matlab_script", 4,10)
    #image_classification("kmeans.txt")

if __name__ == "__main__":
    main()

import scipy.misc
import os
import matplotlib.image as mpimg
import cv2
import numpy as np
import random
import math


datasetDirectory = './GroceryDataset_part1/ProductImagesFromShelves/'
NO_CLASSES = 11

def getImageFileAndLabels():
	labels = []
	imageFiles = []
	for i  in range(NO_CLASSES):
		images = os.listdir(datasetDirectory + str(i+1))
		images = list(map(lambda img_path: datasetDirectory + str(i+1) + '/' + img_path , images))
		imageFiles.extend(images)
		labels.extend([i]*len(images))
	
	return imageFiles, labels 

#This function crops the image into 224x224 size. This is done because the VGG model requires the image to be resized.
def crop_image(x, target_height=224, target_width=224):
    # print(x)
    # p = Path(str(x))
    # image = cv2.imread(str(p.resolve()))
    # print(x)
    image = mpimg.imread(x)
    return cv2.resize(image, (target_height, target_width))
    # if len(image.shape) == 2:
    #     image = np.tile(image[:,:,None], 3)
    # elif len(image.shape) == 4:
    #     image = image[:,:,:,0]

    # height, width, rgb = image.shape
    # if width == height:
    #     resized_image = cv2.resize(image, (target_height,target_width))

    # elif height < width:
    #     resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
    #     cropping_length = int((resized_image.shape[1] - target_height) / 2)
    #     resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    # else:
    #     resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
    #     cropping_length = int((resized_image.shape[0] - target_width) / 2)
    #     resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    # return cv2.resize(resized_image, (target_height, target_width))

def shuffleDataset(images, labels):

	ind_list = [i for i in range(len(images))]
	random.shuffle(ind_list)
	# print(ind_list)
	images  = images[ind_list]
	labels = labels[ind_list]
	return images, labels


def createDataset():
	imageFiles, labels = getImageFileAndLabels()
	images = []
	for i in range(len(imageFiles)):
		images.append(crop_image(imageFiles[i]))
	
	images, labels = shuffleDataset(np.array(images), np.array(labels))
	totalImages = len(images)
	trainImages = images[:int(0.7*totalImages)]
	trainLabels =labels[:int(0.7*totalImages)]
	print(len(trainLabels))
	valImages = images[int(0.7*totalImages): int(0.85*totalImages)]
	valLabels = labels[int(0.7*totalImages): int(0.85*totalImages)]
	print(len(valLabels))
	testImages = images[int(0.85*totalImages):]
	testLabels = labels[int(0.85*totalImages):]
	print(len(testLabels))
	np.save('producttrainNormalImages.npy',trainImages)
	np.save('producttrainNormalLabels.npy',trainLabels)
	np.save('producttestNormalImages.npy',testImages)
	np.save('producttestNormalLabels.npy',testLabels)
	np.save('productvalNormalImages.npy',valImages)
	np.save('productvalNormalLabels.npy',valLabels)
	
	# return images, labels

createDataset()




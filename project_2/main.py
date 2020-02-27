import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *
import pickle

def main():
	#flag for debugging
	flag_subset = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	#data configurations
	pos_data_dir = 'newface16/newface16'
	neg_data_dir = 'nonface16/nonface16'
    
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	
	#compute integral image
	data = integrate_images(normalize(data))

	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = -1 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)
    
	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs,num_bins, drawer, num_cores, boosting_type)
    
	
    #calculate filter values for all training images
	start = time.clock()
	
	#calculating activations by applying the Haar filters to the integral images.
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	#Calling the training function for error calculationa nd weight updation
	boost.train(chosen_wc_cache_dir)

	#visualisation of results: Plotting of Histograms, ROC curves, graphs
	boost.visualize()
	
	#Face Detection for 3 test images
	for i in range(3):
		original_img = cv2.imread('Face_%d.jpg' %i, cv2.IMREAD_GRAYSCALE)
		result_img = boost.face_detection(original_img)
		cv2.imwrite('Result_img_%s_%d.png' % boosting_type, %i, result_img)


    #hard negative mining
	for i in range(3):
		hard_neg_data_dir='Hard_neg_data_%d.pkl' %i
		hard_neg_label_dir='Hard_neg_labels_%d.pkl' %i
		#load hard negatives into original data and retrain
		if hard_neg_data_dir is not None and os.path.exists(hard_neg_data_dir):
			print('[Find cached hard negative data %s loading...]' % hard_neg_data_dir)
			patches=pickle.load(open(hard_neg_data_dir, 'rb'))
			hard_neg_labels=pickle.load(open(hard_neg_label_dir,'rb'))
			hard_neg_labels=hard_neg_labels[:,0,0]
			boost.data=np.concatenate((boost.data,patches), axis=0)
			boost.labels=np.concatenate((boost.labels, hard_neg_labels), axis=0)
		#compute hard negatives from non-face images and store result in pickle file
		else:
			back_img = cv2.imread('Non_face_%d.jpg' %(i+1), cv2.IMREAD_GRAYSCALE)
			patches = boost.get_hard_negative_patches( back_img,scale_step = 10)
			patches=patches[0]
			hard_labels=np.full(boost.labels.shape, fill_value=-1)
			pickle.dump(patches, open('Hard_neg_data_%d.pkl' %i, 'wb'))
			pickle.dump(hard_labels,open('Hard_neg_labels_%d.pkl' %i, 'wb'))
	

	#####################################################	
	

if __name__ == '__main__':
	main()

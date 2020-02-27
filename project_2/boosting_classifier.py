import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import copy
import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None
		
		#create different classifiers depending on the Boosting style
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	#Calculating training activations of HaarFilters applied on the integral images
	def calculate_training_activations(self, save_dir, load_dir):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	
	
	
	
	def train(self, save_dir = None):
		######################
		######## TODO ########

		#Adaboost training

		if self.style == 'Ada':
			if save_dir is not None and os.path.exists(save_dir):
				print('[Find cached training data %s loading...]' % save_dir)
				self.load_trained_wcs(save_dir)
			else:
				m=self.data.shape[0]
				data_weights=np.ones(m)*1/m            #uniform distribution initialisation
				self.chosen_wcs=[]
				self.strong_classifier_error=[]
				for i in range(110):
					error_list=[]
					for j in range(len(self.weak_classifiers)):
						#calculating error for each weak classifier and storing it
						error_temp=self.weak_classifiers[j].calc_error(data_weights,self.labels)  
						error_list.append(error_temp)

					#selecting classifier with minimum error and computing alpha
					min_error_index=np.argmin(error_list)
					min_error=error_list[min_error_index]
					alpha=0.5*(np.log((1-min_error)/min_error))
					print(min_error, min_error_index, alpha)
					
					h_x=np.zeros(data_weights.shape[0])
					for j in range(data_weights.shape[0]):
						h_x[j]=self.weak_classifiers[min_error_index].predict_image(self.data[j])
						data_weights[j]=data_weights[j]*np.exp(-self.labels[j]*h_x[j]*alpha)
					data_weights=data_weights/np.sum(data_weights)              #normalise weights
					
					#select weak classifiers to form a strong classifier
					self.chosen_wcs.append((alpha,copy.deepcopy(self.weak_classifiers[min_error_index]))) 
					### deepcopy is used to avoid change of value since pointer is stored.
					
					#cache training results to self.visualizer for visualization
					score=[]
					for img in self.data:
						score.append(self.sc_function(img))
					accuracy=np.mean(np.sign(score) == self.labels)
					self.visualizer.strong_classifier_error.append(1- accuracy)
					
					error_list=np.asarray(error_list, dtype=np.float64)	
					
					if i in [0,9,49,99]:
						self.visualizer.strong_classifier_scores[i+1]=score
						self.visualizer.weak_classifier_accuracies[i+1]= sorted(1- error_list)

			

		######################
			#cache chosen_wcs for later calculation of realboost classifiers
			if save_dir is not None:
				pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))
		
		

		#Realboost training
		elif self.style == 'Real':
			#load training activations from Adaboost
			print('[Find cached training data %s loading...]' % save_dir)
			weak_classifier_pool=pickle.load(open(save_dir, 'rb'))	
			
			self.chosen_wcs=[]
			m=self.data.shape[0]
			data_weights=np.ones(m)*1/m       #weights initialised to uniform distribution

			for i in range(110):
				error=self.weak_classifiers[weak_classifier_pool[i][1].id].calc_error(data_weights,self.labels) #calc_error called referencing the realboost object with id
				h_x=np.zeros(data_weights.shape[0])
				alpha=1                       #since alpha is absorbed by h_x in RealBoost
				for j in range(data_weights.shape[0]):
					h_x[j]=self.weak_classifiers[weak_classifier_pool[i][1].id].predict_image(self.data[j])
					data_weights[j]=data_weights[j]*np.exp(-self.labels[j]*h_x[j]*alpha)   #weight updation
				data_weights=data_weights/np.sum(data_weights)
				self.chosen_wcs.append((alpha,copy.deepcopy(self.weak_classifiers[weak_classifier_pool[i][1].id])))

				#cache training results to self.visualizer for visualization
				score=[]
				for img in self.data:
					score.append(self.sc_function(img))
				if i in [10,50,100]:
					self.visualizer.strong_classifier_scores[i+1]=score

		#########################



			
	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, scale_step = 20):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		
		
		##################################################################################

		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3

		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		wrong_patches = patches[np.where(np.array(predicts) > 0), ...]
		print(wrong_patches.shape)
		return wrong_patches

	def visualize(self):
		self.visualizer.labels = self.labels
		print(len(self.chosen_wcs))
		for i in range(20):
			self.visualizer.print_haar_filter(self.chosen_wcs[i][1],i) #print the top 20 Haar Filters
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		self.visualizer.draw_wc_accuracies()
		self.visualizer.draw_strong_classifier_error()

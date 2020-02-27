import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import cv2

class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.labels = None
		self.strong_classifier_error=[]

	def draw_histograms(self):
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
			plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers' % t)
			plt.savefig('histogram_%d.png' % t)

	def draw_rocs(self):
		plt.figure()
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve')

	def draw_wc_accuracies(self):
		plt.figure()
		for t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[t]
			plt.plot(accuracies, label = 'After %d Selection' % t)
		plt.ylabel('Accuracy')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Accuracies')
		plt.legend(loc = 'upper right')
		plt.savefig('Weak Classifier Accuracies')

	#get chosen weak classifier object and number of filters to be printed
	def print_haar_filter(self,weak_classifier,i):
		img=np.full((32,32),fill_value=150).astype(np.uint8).copy() #initialising grey background
		for rect in weak_classifier.plus_rects: 
			cv2.rectangle(img,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])), (255,255,255), thickness=-1, lineType=8, shift=0)
		for rect in weak_classifier.minus_rects:
			cv2.rectangle(img,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])), (0,0,0) , thickness=-1, lineType=8, shift=0)
		cv2.imshow('HaarFilter_%d' %i,img)
		cv2.imwrite('HaarFilter_%d.jpg' %i, img)

	def draw_strong_classifier_error(self):
		#print(self.strong_classifier_error)
		plt.figure()
		plt.plot(self.strong_classifier_error)
		plt.ylabel('Error Rate')
		plt.xlabel('Epoch')
		plt.title('Top 100 Strong Classifier Error Rate')
		plt.legend(loc = 'upper right')
		plt.savefig('Strong Classifier Error')

if __name__ == '__main__':
s	main()

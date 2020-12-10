package main;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import dataset.*;
import eval.ConfusionMatrix;
import eval.CrossValEntry;
import eval.Evaluation;
import classifiers.AbstractClassifier;
import classifiers.Centroid;
import classifiers.KmeansClassifier;
import classifiers.KnnClassifier;

public class Main {

	public static void main(String[] args) throws IOException {
		// DEMO
		System.out.println("Working Directory = " + System.getProperty("user.dir"));
		
		
//		Demo.img();
//		Demo.dataset();
//		Demo.split_dataset_into_train_test();
//		Demo.knn_classifier();
//		Demo.kmeans_classifier();
//		Demo.cross_val();
//		Demo.leave_one_out();
//		Demo.compare_multiple_classifiers_with_train_test_split();
//		Demo.compare_multiple_classifiers_with_k_fold();
//		Demo.confusion_matrix();
	}
	
	static class Demo {
		/*
		 * Class that will contains examples and quick demos to get familiar with the code
		 */
		
		public static void img() {
			/* 
			 * Image related examples
			 */
			
			// 1st way
			
			// define path
			String path = "project_files/E34/s01n001.E34";
			// create the corresponding representation
			
			try {
				Representation r = new Representation(path);
				// create the img
				Image img = new Image(r); // this constructor will try to conclude the label from the file name
				
				// otherwise, you can specify the label explicitly
				int label = 1;
				Image img1 = new Image(r, label);
			} catch (BadRepresentationFileException e) {
				e.printStackTrace();
			} 
			
			// 2nd way, easier
			
			try {
				Image img2 = new Image(new Representation("project_files/E34/s01n001.E34"));
			} catch (BadRepresentationFileException e) {
				e.printStackTrace();
			}
			
			// create a custom img with an array of values
			
			Double[] d1 = {1.2, 3.4};
			Image img1 = new Image(new Representation(d1, "representation_type"));
			img1.set_label(1);
		
			// get infos about the img 
			
			try {
				Image img = new Image(new Representation(path));
				
				System.out.println(img.get_class()); // animal, outil.....
				System.out.println(img.get_label()); // 1,2,...,9
				System.out.println(img.get_representation_type()); // E34, SA, GFD or F0
				System.out.println(img.get_coords()); // the values of the corresponding representation
			} catch (BadRepresentationFileException e) {
				e.printStackTrace();
			}
			
		}
		
		public static void dataset() {
			/*
			 * dataset related examples.
			 */
			
			// from a folder that contains multiple images.
			try {
				Dataset dataset = new Dataset("project_files/GFD");
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			// from a list of imgs
			
			try {
				Dataset dataset = new Dataset();
				List<Image> imgs = new ArrayList<>();
				
				imgs.add(new Image(new Representation("project_files/E34/s01n001.E34")));
				imgs.add(new Image(new Representation("project_files/E34/s02n002.E34")));
				imgs.add(new Image(new Representation("project_files/E34/s03n003.E34")));
				
				for (Image img : imgs)
					dataset.add_datapoint(img, img.get_label()); // or specify the label yourself 
					
			} catch (BadRepresentationFileException e) {
				e.printStackTrace();
			}
			
			try {
				Dataset dataset = new Dataset("project_files/GFD");
				
				// 1st way
				for (Image img : dataset.keySet())  
					System.out.println(img);
				
				// 2nd way
				for (Entry<Image, Integer> entry : dataset.entrySet()) {
					// Image = entry.getKey(), Label = entry.getValue()
					System.out.println(entry.getKey());
					System.out.println(entry.getValue());
				}
				
				// if you need all images of a certain class/label
				// example getting all rabbit imgs ( label = 2 ) [ assuming they're in the dataset ]
				for (Image rabbit_img : dataset.get_stratums().get(2)) 
					System.out.println(rabbit_img);
				
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public static void split_dataset_into_train_test() {
			// specify the train percentage
			double train_percentage = 0.7;
			
			try {
				// get the whole dataset
				Dataset dataset = new Dataset("project_files/GFD");
				
				// split it
				List<Dataset> datasets = Evaluation.split_dataset(dataset, train_percentage);
				
				// your training dataset is the first while the test set is the second
				Dataset train_set = datasets.get(0);
				Dataset test_set = datasets.get(1);
				
				System.out.println("The complete dataset is a " + dataset);
				System.out.println("Training set size = " + train_set.size());
				System.out.println("Training set size = " + test_set.size());
				
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public static void knn_classifier() {
			/*
			 * Create a knn classifier, train it and get its accuracy
			 */
			
			// get your training and test datasets, if you have them in a folder otherwise you can 
			// split randomly a dataset using the methode above
			
			// we used this method just to give another use case
			try {
				Dataset training_dataset = new Dataset("project_files/GFD/train");
				Dataset test_dataset = new Dataset("project_files/GFD/test/");	
				
				// Create the KNN
				int k = 3, p = 2; // for example
				
				KnnClassifier knn = new KnnClassifier(k, p);
				System.out.println(knn);
				
				// train it on train_set
				knn.train(training_dataset);
				
				// print the accuracies
				System.out.println("train acc = " + knn.accuracy(training_dataset));
				System.out.println("test acc = " + knn.accuracy(test_dataset));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		public static void kmeans_classifier() {
			/*
			 * Create a variation of kmeans, train it and test it
			 */
			
			try {
				List<Dataset> d = Evaluation.split_dataset(new Dataset("project_files/GFD"), 0.8);
				Dataset training_dataset = d.get(0);
				Dataset test_dataset = d.get(1);	
				
				// Create the absolutely random Kmeans
				int k = 9; // we want to have 9 clusters cuz we know we have 9 classes
				int p = 4; // define which distance we'll use
				long max_iter = 100; // max number of iterations

				KmeansClassifier abs_random_kmeans = new KmeansClassifier(k, p, max_iter, "random");
				System.out.println(abs_random_kmeans);
				
				// train it on train_set
				abs_random_kmeans.train(training_dataset);
				
				// print the accuracies
				System.out.println("train acc = " + abs_random_kmeans.accuracy(training_dataset));
				System.out.println("test acc = " + abs_random_kmeans.accuracy(test_dataset));
				System.out.println();
				
				// enhanced random kmeans ( equivalent to normalizing the dataset )
				KmeansClassifier enhanced_random_kmeans = new KmeansClassifier(k, 1, max_iter, "enhanced_random");
				System.out.println(enhanced_random_kmeans);
				
				// train it on train_set
				enhanced_random_kmeans.train(training_dataset);
				
				// print the accuracies
				System.out.println("train acc = " + enhanced_random_kmeans.accuracy(training_dataset));
				System.out.println("test acc = " + enhanced_random_kmeans.accuracy(test_dataset));
				System.out.println();
				
				// Kmeans++
				// max_iter = 100 and init_method = kmeans++ by default in this constructor
				KmeansClassifier kmeans_plus = new KmeansClassifier(k, 2);
				System.out.println(kmeans_plus);
				
				// train it on train_set
				kmeans_plus.train(training_dataset);	
				
				// print the accuracies
				System.out.println("train acc = " + kmeans_plus.accuracy(training_dataset));
				System.out.println("test acc = " + kmeans_plus.accuracy(test_dataset));
				System.out.println();
				
				// to get the constructed centroids
				System.out.println("constructed centroids: ");
				System.out.println(kmeans_plus.get_clusters().keySet());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public static void cross_val() {
			/*
			 * Example on how to do k fold cross validation
			 */
			
			// Choose a classifier
			KmeansClassifier kmeans = new KmeansClassifier(9, 2);
			
			// load the dataset
			try {
				Dataset dataset = new Dataset("project_files/E34");
				System.out.println(dataset);
				// perform cross val and print infos
				// choose your k
				int k = 4; 
				
				// true -> re shuffle, true -> verbose
				Evaluation.cross_validation(kmeans, dataset, k, true, true);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		public static void leave_one_out() {
			/*
			 * Example on how to perform leave one out
			 */
			
			// Choose a classifier
			KnnClassifier knn = new KnnClassifier(9, 4);
			
			try {
				// load the dataset
				Dataset dataset = new Dataset("project_files/E34");
				System.out.println(dataset);
				
				// perform cross val and print infos
				// k must be = dataset.size(), or you can simplify pick any number that is >= dataset.size()
				int k = dataset.size(); 
				
				// true -> re shuffle, true -> verbose
				Evaluation.cross_validation(knn, dataset, k, true, true);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		public static void compare_multiple_classifiers_with_train_test_split() {
			/*
			 * Example on how to compare multiple classifiers using train test split
			 */
			
			try {
				// load the dataset
				Dataset dataset = new Dataset("project_files/E34");
				System.out.println(dataset);
				
				// create the list of classifiers, must use the type AbstractClassifier, to gather the different
				// classifiers in one list
				
				List<AbstractClassifier> classifiers_list = new ArrayList<>();
				
				// for example, we'll compare knn, Kmeans and Kmeans++
				classifiers_list.add(new KnnClassifier(3, 3));
				classifiers_list.add(new KmeansClassifier(9, 3, 100, "enhanced_random"));
				classifiers_list.add(new KmeansClassifier(9, 3));
				
				// define your training ration
				double train_percentage = 0.7;
				
				// want details ?
				boolean verbose = true;
				
				// save accuracies into a list
				List<Double> accuracies = Evaluation.train_test_split_accuracy(classifiers_list, dataset, train_percentage, verbose);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		public static void compare_multiple_classifiers_with_k_fold() {
			/*
			 * Example on how to compare multiple classifiers using cross validation
			 */
			
			try {
				// load the dataset
				Dataset dataset = new Dataset("project_files/E34");
				System.out.println(dataset);
				
				// create the list of classifiers, must use the type AbstractClassifier, to gather the different
				// classifiers in one list
				
				List<AbstractClassifier> classifiers_list = new ArrayList<>();
				
				// for example, we'll compare knn, Kmeans and Kmeans++
				classifiers_list.add(new KnnClassifier(1, 1));
				classifiers_list.add(new KmeansClassifier(9, 4, 100, "enhanced_random"));
				classifiers_list.add(new KmeansClassifier(9, 2));
				
				// define your k value
				int k = 3; // if you want to perform leave one out -> k = dataset.size()
				
				// want details ?
				boolean verbose = true, shuffle = true;
				
				// save accuracies into a list
				List<CrossValEntry> accuracies = Evaluation.cross_validation(classifiers_list, dataset, k, shuffle, verbose);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		public static void confusion_matrix() {
			/*
			 * Example on to get the confusion matrix and its related metrics, of a classifier
			 */
			
			// Choose a classifier
			KnnClassifier classifier = new KnnClassifier(1, 1);
			
			// load the dataset
			try {
				Dataset dataset = new Dataset("project_files/F0");
				System.out.println(dataset);
				
				// split it, so he can train on the first and be evaluated on the second
				List<Dataset> d = Evaluation.split_dataset(dataset, 0.7);
				
				// create the confusion matrix
				ConfusionMatrix matrix = Evaluation.confusion_matrix(classifier, d.get(0), d.get(1));
				
				// print it
				System.out.println(matrix);
				
				// calculate a precision of a given label
				System.out.println("Precision of label 3 = " + matrix.get_precision(3));
				
				// calculate a recall of a certain label
				System.out.println("Recall of label 1 = " + matrix.get_precision(1));
				
				// calculate the global precision and recall
				System.out.println("Precision = " + matrix.get_precision() + " while Recall = " + matrix.get_recall());
				
				// calculate the f1-score
				System.out.println("F1-score = " + matrix.get_f1_score());
				
				// calculate a certain F score
				System.out.println("F-score with beta = 0.5, is " + matrix.get_f_score(0.5));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	private static void demo() {
		Double[] d1 = {1.2, 3.4};
		Image img1 = new Image(new Representation(d1, "img1"));
		img1.set_label(1);
		
		Double[] d2 = {1.98, 9.4};
		Image img2 = new Image(new Representation(d2, "img1"));
		img2.set_label(1);
		
		Double[] d3 = {20.455, 34.4};
		Image img3 = new Image(new Representation(d3, "img1"));
		img3.set_label(2);

		Double[] d4 = {14.56, 13.4};
		Image img4 = new Image(new Representation(d4, "img1"));
		img4.set_label(2);
		
		Double[] d5 = {200.455, 34.4};
		Image img5 = new Image(new Representation(d5, "img1"));
		img5.set_label(3);

		Double[] d6 = {180.2, 13.4};
		Image img6 = new Image(new Representation(d6, "img1"));
		img6.set_label(3);
		
		
		Image[] imgs = {img1, img2, img3, img4, img5, img6};
		Dataset d = new Dataset();
		
		for (Image img : imgs) {
			d.add_datapoint(img, img.get_label());
		}
		
		KmeansClassifier kmeans = new KmeansClassifier(3, 2, 100, "kmeans++");
		kmeans.train(d);
		System.out.println(kmeans.accuracy(d));
		System.out.println(kmeans.get_clusters().keySet());
		
		for (Image i : imgs) 
			System.out.println("Image label = " + i.get_label() + " -> predicted " + kmeans.predict(i));
		
		System.out.println("Image label = " + img4.get_label() + " -> predicted " + kmeans.predict(img4));
	}

}

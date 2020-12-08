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
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import dataset.*;
import eval.ConfusionMatrix;
import eval.Evaluation;
import classifiers.AbstractClassifier;
import classifiers.Centroid;
import classifiers.KmeansClassifier;
import classifiers.KnnClassifier;

public class Main {

	public static void main(String[] args) throws IOException {
		// DEMO
		System.out.println("Working Directory = " + System.getProperty("user.dir"));
				
		Dataset training_dataset = new Dataset("project_files/GFD/");
//		Dataset test_dataset = new Dataset("project_files/GFD/test/");	
		
//		AbstractClassifier kmeans = new KmeansClassifier(5, 2, 100, "random");
//		kmeans.train(training_dataset);
//		
//		System.out.println(kmeans instanceof AbstractClassifier);
//		System.out.println(kmeans.accuracy(training_dataset));
//		System.out.println(kmeans.get_clusters().keySet());
		
		AbstractClassifier knn = new KnnClassifier(3);
		AbstractClassifier en_rand_kmeans = new KmeansClassifier(9, 2, 100, "enhanced_random");
		AbstractClassifier random_kmeans = new KmeansClassifier(9, 2, 100, "random");
		AbstractClassifier kmeans = new KmeansClassifier(9, 2, 100, "kmeans++");
		
		Evaluation.train_test_split_accuracy(Arrays.asList(new AbstractClassifier[] {kmeans, en_rand_kmeans, random_kmeans}), training_dataset, 0.7, true);
		
//		List<Dataset> datasets = Evaluation.split_dataset(training_dataset, 0.7);
//		
//		System.out.println(datasets.get(0).size());
//		System.out.println(datasets.get(1).size());
//		
//		System.out.println("\ndistinct labels in training set = " + datasets.get(0).get_stratums().keySet().size());
//		System.out.println("distinct labels in test set = " + datasets.get(1).get_stratums().keySet().size());
//		System.out.println();
//		
//		
//		System.out.println("train set");
//		for( int i : datasets.get(0).get_stratums().keySet() ) 
//			System.out.println("class " + i + " has " + datasets.get(0).get_stratums().get(i).size() + " instances");
//		
//		System.out.println("\ntest set");
//		for( int i : datasets.get(1).get_stratums().keySet() ) 
//			System.out.println("class " + i + " has " + datasets.get(1).get_stratums().get(i).size() + " instances");
//		
//		System.out.println(datasets.get(0).get_dataset());
//		System.out.println(datasets.get(1).get_dataset());
		
//		Evaluation.accuracy(random_kmeans, training_dataset, training_dataset);
//		Evaluation.accuracy(Arrays.asList(new AbstractClassifier[] {kmeans, en_rand_kmeans, random_kmeans}), training_dataset, training_dataset);
//		Evaluation.cross_validation(kmeans, training_dataset, 1005, true);
//		Evaluation.cross_validation(Arrays.asList(new AbstractClassifier[] {knn, kmeans}), training_dataset, 4, true, true);
		
//		ConfusionMatrix c = Evaluation.confusion_matrix(knn, training_dataset, training_dataset);
//		System.out.println(c);
//		System.out.println(c.get_accuracy());
//		System.out.println();
//		System.out.println(c.get_recall());
//		System.out.println(c.get_precision());
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

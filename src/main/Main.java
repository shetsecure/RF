package main;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
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
		
		Dataset training_dataset = new Dataset("project_files/E34/kmeans/");
//		Dataset test_dataset = new Dataset("project_files/GFD/test/");	
		
//		AbstractClassifier kmeans = new KmeansClassifier(5, 2, 100, "random");
//		kmeans.train(training_dataset);
//		
//		System.out.println(kmeans instanceof AbstractClassifier);
//		System.out.println(kmeans.accuracy(training_dataset));
//		System.out.println(kmeans.get_clusters().keySet());
		
		AbstractClassifier knn = new KnnClassifier(3);
		AbstractClassifier kmeans = new KmeansClassifier(9, 2);
//		Evaluation.cross_validation(Arrays.asList(new AbstractClassifier[] {knn, kmeans}), training_dataset, 4, true, true);
		
		ConfusionMatrix c = Evaluation.confusion_matrix(knn, training_dataset, training_dataset);
		System.out.println(c);
		System.out.println(c.get_accuracy());
		
//		int group_size = 5;
//		
//		List<Integer> all_indices = IntStream.range(0, 21).boxed().collect(Collectors.toList());
//		
//		// constructing a list of indices
//		List<List<Integer>> k_index_lists = new ArrayList<>();
//		
//		for (int i = 0; i < 4; i++) {
//			// shuffle the indices list
//			Collections.shuffle(all_indices);
//			
//			k_index_lists.add(all_indices.stream().limit(group_size).collect(Collectors.toList()));
//			all_indices.removeAll(all_indices.stream().limit(group_size).collect(Collectors.toList()));
//		}
//		System.out.println(k_index_lists);
//		System.out.println(k_index_lists.get(3));
//		
////		System.out.println(k_index_lists.stream().flatMap(List::stream).collect(Collectors.toList()));
//		System.out.println(k_index_lists.stream().flatMap(List::stream).filter(i -> !k_index_lists.get(3).contains(i)).collect(Collectors.toList()));
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

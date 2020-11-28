package main;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import dataset.*;
import classifiers.Centroid;
import classifiers.Kmeans;
import classifiers.KmeansClassifier;

public class Main {

	public static void main(String[] args) throws IOException {
		// DEMO
		//System.out.println("Working Directory = " + System.getProperty("user.dir"));
		
		Dataset training_dataset = new Dataset("project_files/E34/kmeans/");
		Dataset test_dataset = new Dataset("project_files/E34/test/");
//			
//		Kmeans kmeans = new Kmeans(4, 2, 100, "++");
//		kmeans.train(training_dataset);
//		System.out.println(kmeans.accuracy(training_dataset));
//		System.out.println(kmeans.clusters.keySet());

		KmeansClassifier kmeans1 = new KmeansClassifier(4);
		kmeans1.train(training_dataset);		
		System.out.println(kmeans1.accuracy(training_dataset));
		System.out.println(kmeans1.centroids_map.values());
		System.out.println(kmeans1.centroids_map.values().stream().distinct().collect(Collectors.toList()).size());
		
//		demo();
		
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
		
		Kmeans kmeans = new Kmeans(3, 2, 10000, "++");
		kmeans.train(d);
		System.out.println(kmeans.accuracy(d));
		System.out.println(kmeans.clusters.keySet());
		
		for (Image i : imgs) 
			System.out.println("Image label = " + i.get_label() + " -> predicted " + kmeans.predict(i));
		
		System.out.println("Image label = " + img4.get_label() + " -> predicted " + kmeans.predict(img4));
		
		KmeansClassifier kmeans_demo = new KmeansClassifier(2);
		kmeans_demo.train(d);
		System.out.println(kmeans_demo.accuracy(d));
		System.out.println(kmeans_demo.centroids_map);
	}

}

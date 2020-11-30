package main;
import java.io.IOException;
import java.util.stream.Collectors;

import dataset.*;
import classifiers.Centroid;
import classifiers.Kmeans;

public class Main {

	public static void main(String[] args) throws IOException {
		// DEMO
		//System.out.println("Working Directory = " + System.getProperty("user.dir"));
		
		Dataset training_dataset = new Dataset("project_files/GFD/train/");
		Dataset test_dataset = new Dataset("project_files/GFD/test/");	
		
		Kmeans kmeans = new Kmeans(9, 2, 100, "kmeans++");
		kmeans.train(training_dataset);
		System.out.println(kmeans.clusters.keySet());
		System.out.println("Distinct labels: " + kmeans.clusters.keySet().stream().map(Centroid::get_label)
														.distinct().collect(Collectors.toList()).size());
		
		System.out.println(kmeans.accuracy(training_dataset));
		System.out.println(kmeans.accuracy(test_dataset));
		
//		try {
//			Image img = new Image(new Representation("/home/shetsecure/eclipse-workspace/RF/project_files/E34/kmeans/s03n002.E34"));
//			Dataset d = new Dataset();
//			d.add_datapoint(img, 3);
//			
//			System.out.println(kmeans1.accuracy(d));
//		} catch (BadRepresentationFileException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
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
		
		Kmeans kmeans = new Kmeans(3, 2, 100, "kmeans++");
		kmeans.train(d);
		System.out.println(kmeans.accuracy(d));
		System.out.println(kmeans.clusters.keySet());
		
		for (Image i : imgs) 
			System.out.println("Image label = " + i.get_label() + " -> predicted " + kmeans.predict(i));
		
		System.out.println("Image label = " + img4.get_label() + " -> predicted " + kmeans.predict(img4));
	}

}

package classifiers;
import java.util.List;
import java.util.Map.Entry;

import dataset.Dataset;
import dataset.Image;

import dataset.*;

public abstract class AbstractClassifier {
	/* Abstract class that will serve as a blueprint to all classifiers
	 * Each classifier needs to be trained on the dataset
	 * Each classifier must have a predict method
	 * Most classifiers have a some kind of scoring method to calculate in order to classify
	
	 * accuracy is the same for all of them, so it is implemented here.
	 *  
	 */
	
	protected abstract boolean train(Dataset training_dataset);
	
	protected abstract int predict(Image img);
	
	@Override
	public abstract String toString();
	
	public double accuracy(Dataset test_dataset) {
		int counter = 0, size = test_dataset.size();
		
		for(Entry<Image, Integer> entry : test_dataset.entrySet()) {
			int predicted_label = predict(entry.getKey());
			
			if ( predicted_label == entry.getValue())
				counter++;
//			else {
//				System.err.println(entry.getKey());
//				System.err.println("Classifier predicted: " + predicted_label 
//						+ "The real label was " + entry.getKey().get_label());
//			}
		}
		
		// casting counter to double to have a floating point division
		return (double) counter / size;
	}
	
	protected double dist(double[] x, double[] y, int p) {
		// generally for debugging purposes
		assert p > 0;
		
		if (p == 1)
			return manhattan_dist(x, y);
		else if (p == 2)
			return euclidean_dist(x, y);
		
		return minkowski_dist(x, y, p);
	}
	
	// Overloading of the method above to make things simpler
	
	protected double dist(List<Double> l_x, List<Double> l_y, int p) {
		// generally for debugging purposes
		
		double[] x = convert_list_to_double(l_x);
		double[] y = convert_list_to_double(l_y);
		
		return dist(x, y, p);
	}
	
	protected double dist(Representation r_x, Representation r_y, int p) {	
		// will be used in production
		return dist(r_x.get_data(), r_y.get_data(), p);
	}
	
	protected double dist(Image img_x, Image img_y, int p) {
		// will be used in production
		return dist(img_x.get_representation(), img_y.get_representation(), p);
	}
	
	protected double euclidean_dist(double[] x, double[] y) {
		assert x.length == y.length;
		
		double sum = 0;
		
		for (int i = 0; i < x.length; i++)
			sum += Math.pow((x[i] - y[i]), 2);
		
		return Math.sqrt(sum);
	}
	
	protected double manhattan_dist(double[] x , double[] y) {
		assert x.length > 0 && x.length == y.length;
		
		double sum = 0;
		
		for (int i = 0; i < x.length; i++)
			sum += Math.abs((x[i] - y[i]));
		
		return sum;
	}
	
	protected double minkowski_dist(double[] x, double[] y, int p) {
		assert p > 0;
		assert x.length > 0 && x.length == y.length;
		
		double sum = 0;
		
		for (int i = 0; i < x.length; i++)
			sum += Math.pow((x[i] - y[i]), p);
		
		return Math.pow(sum, 1.0/p);
	}
	
	protected double[] convert_list_to_double(List<Double> l) {
		boolean null_exists = l.stream().anyMatch(d -> d == null);
		
		if (null_exists) 
			System.err.println("Warning: Null exists in the passed list, will be replaced by 0...");
		
		double[] arr = l.stream().map(i -> (i == null ? 0 : i))
								 .mapToDouble(Double::doubleValue)
								 .toArray();
		
		return arr;
	}
}

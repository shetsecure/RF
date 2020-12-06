package eval;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ConfusionMatrix {
	private double[][] data; // lines are the true labels, where cols are the predicted labels
	private int dim;
	private boolean normalized;
	private List<Integer> distinct_labels;
	
	private ConfusionMatrix() {}
	
	public ConfusionMatrix(List<Integer> true_labels, List<Integer> predicted_labels) {
		assert true_labels != null && predicted_labels != null;
		assert true_labels.size() > 0 && true_labels.size()== predicted_labels.size();
		
		// get the number of distinct elements from the two lists
		distinct_labels = Stream.concat(true_labels.stream(), predicted_labels.stream()).distinct().collect(Collectors.toList());
		dim = distinct_labels.size();
		
		dim += 1; // for the time being, cuz labels starts with 1
		data = new double[dim][dim];
		
		normalized = false;
		
		for (int i = 0; i < true_labels.size(); i++) 
			data[true_labels.get(i)][predicted_labels.get(i)] += 1;
	}
	
	public void normalize() {
		normalized = true;
	}
	
	public double[][] get_matrix() {
		return data;
	}
	
	public double get_accuracy() {
		double diagonal_sum = IntStream.range(0, dim).asDoubleStream().map(i -> data[(int) i][(int) i]).sum();
		double sum_all_elements = Arrays.stream(data).flatMapToDouble(Arrays::stream).sum(); 
		
		return diagonal_sum / sum_all_elements;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		int i = 0;
	    for(double[] row : data) {
	    	if (i++ == 0) continue;
	    	sb.append(Arrays.toString(Arrays.copyOfRange(row, 1, dim)));
	    	sb.append("\n");
	    }

	    return sb.toString();
	}
}

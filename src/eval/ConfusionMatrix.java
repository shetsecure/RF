package eval;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ConfusionMatrix {
	private double[][] data; // lines are the true labels, where cols are the predicted labels
	private int dim;
	private List<Integer> distinct_labels;
	
	@SuppressWarnings("unused")
	private ConfusionMatrix() {}
	
	public ConfusionMatrix(List<Integer> true_labels, List<Integer> predicted_labels) {
		assert true_labels != null && predicted_labels != null;
		assert true_labels.size() > 0 && true_labels.size()== predicted_labels.size();
		
		// get the number of distinct elements from the two lists
		distinct_labels = Stream.concat(true_labels.stream(), predicted_labels.stream()).distinct().collect(Collectors.toList());
		dim = distinct_labels.size();
		
		dim += 1; // for the time being, cuz labels starts with 1
		data = new double[dim][dim];
		
		for (int i = 0; i < true_labels.size(); i++) 
			data[true_labels.get(i)][predicted_labels.get(i)] += 1;
		
//		System.out.println(true_labels);
//		System.out.println(predicted_labels);
	}

	public double[][] get_matrix() {
		return data;
	}
	
	public double get_accuracy() {
		double diagonal_sum = IntStream.range(0, dim).asDoubleStream().map(i -> data[(int) i][(int) i]).sum();
		double sum_all_elements = Arrays.stream(data).flatMapToDouble(Arrays::stream).sum(); 
		
		return diagonal_sum / sum_all_elements;
	}
	
	public double get_recall() {	
		return IntStream.range(0, distinct_labels.size()).mapToDouble(i -> get_recall(distinct_labels.get(i))).average().orElse(0.0);
	}
	
	public double get_recall(int label) {
		assert distinct_labels.contains(label);
		
		double vp_i = data[label][label];
		double fn_i = Arrays.stream(data[label]).sum() - vp_i; // sum of all cols of the line_i - vp_i
		
		return vp_i / (vp_i + fn_i);
	}
	
	public double get_precision() {
		return IntStream.range(0, distinct_labels.size()).mapToDouble(i -> get_precision(distinct_labels.get(i))).average().orElse(0.0);
	}
	
	public double get_precision(int label) {
		assert distinct_labels.contains(label);
		
		double vp_i = data[label][label];
		double fp_i = IntStream.range(0, data.length).mapToDouble(i -> data[i][label]).sum() - vp_i; // sum of column label
		
		return vp_i / (vp_i + fp_i);
	}
	
	public double get_f_score(double beta) {
		double precision = get_precision();
		double recall = get_recall();
		double beta_sq = beta * beta;
		
		return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall);
	}
	
	public double get_f1_score() {
		return get_f_score(1.0);
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

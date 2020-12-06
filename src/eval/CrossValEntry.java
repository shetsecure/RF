package eval;

import java.util.AbstractMap;
import java.util.Map;

public class CrossValEntry {
	/*
	 * Helper class used in cross_validation method in Evaluation class.
	 */
	final double avg_training_accuracy, training_std_dev;
	final double avg_test_accuracy, test_std_dev;
	
	public CrossValEntry(double train_acc, double train_std, double test_acc, double test_std) {
		avg_training_accuracy = train_acc;
		training_std_dev = train_std;
		avg_test_accuracy = test_acc;
		test_std_dev = test_std;
	}
	
	public double get_training_acc() {
		return avg_training_accuracy;
	}
	
	public double get_test_acc() {
		return avg_test_accuracy;
	}
	
	public double get_training_std() {
		return training_std_dev;
	}
	
	public double get_test_std() {
		return test_std_dev;
	}
	
	public Map.Entry<Double, Double> get_training_info() {
		return new AbstractMap.SimpleImmutableEntry<>(avg_training_accuracy, training_std_dev);
	}
	
	public Map.Entry<Double, Double> get_test_info() {
		return new AbstractMap.SimpleImmutableEntry<>(avg_test_accuracy, test_std_dev);
	}
	
	@Override
	public String toString() {
		return "avg_training_accuracy : " + avg_training_accuracy + " with an std_dev of : " + training_std_dev + "\n"
			 + "avg_test_accuracy : " + avg_test_accuracy + " with an std_dev of : " + test_std_dev + "\n";
	}
}

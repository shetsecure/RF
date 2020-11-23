import java.util.*;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {

	public static void main(String[] args) {
		// DEMO

		Dataset training_dataset = new Dataset("project_files/E34/train/");
		Dataset test_dataset = new Dataset("project_files/E34/test/");
		
		System.out.println("Training dataset size : " + training_dataset.size());
		System.out.println("Testing dataset size : " + test_dataset.size());
		
		int k = 1; // k = 1, will always result in a 100% training accuracy
		int p = 1; // Minkowski metric parameter
		KnnClassifier knn = new KnnClassifier(k, p);
		
		knn.train(training_dataset);
		
		System.out.println("Using " + knn);
		
		System.out.println("Training accuracy: " + knn.accuracy(training_dataset) * 100 + "%");
		
		// horrible results for now...
		System.out.println("Testing accuracy: " + knn.accuracy(test_dataset) * 100 + "%");

		
		
	}

}

package eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import classifiers.AbstractClassifier;
import dataset.Dataset;
import dataset.Image;

public final class Evaluation {
	// Static class, doesn't make any sense to instantiate it
	
	private Evaluation() {}
	
	/*
	 * Comparing a list of classifiers with K-fold cross validation (totally Random), Stratified version below.
	 */
	
	public static List<CrossValEntry> cross_validation(List<AbstractClassifier> classifiers, Dataset dataset, int k, boolean verbose, boolean shuffle) {
		assert classifiers.size() > 0;
		for (int i = 0; i < classifiers.size(); i++)
			assert classifiers.get(i) != null;
			
		assert k > 1;	
		
		if (shuffle)
			dataset.shuffle();
		
		boolean leave_one_out = k >= dataset.size();
		
		List<Image> datapoints = dataset.keySet().stream().collect(Collectors.toList()); // keeping references to instances
		List<CrossValEntry> results = new ArrayList<>(); // list of results that will be returned
		
		List<List<Double>> training_accs = new ArrayList<>(); // list where we keep training accuracies for each classifier
		List<List<Double>> test_accs = new ArrayList<>(); // list where we keep training accuracies for each classifier
		
		for (int i = 0; i < classifiers.size(); i++) {
			training_accs.add(new ArrayList<>());
			test_accs.add(new ArrayList<>());
		}
		
		if (leave_one_out) {
			// pick one instance for test, the rest for training
			int choice = 0, corresponding_label; 
			Image picked_instance;
			
			for (int i = 0; i < datapoints.size(); i++) {
				// pick one for test, remove it from dataset so we can do our training
				picked_instance = datapoints.get(choice);
				corresponding_label = dataset.get(picked_instance);
				dataset.remove_datapoint(picked_instance);
				
				for (int j = 0; j < classifiers.size(); j++) {
					// training each classifier on the new dataset = (original_dataset - picked_instance)
					classifiers.get(j).train(dataset);
					
					// adding results to each list
					training_accs.get(j).add(classifiers.get(j).accuracy(dataset));
					test_accs.get(j).add( (classifiers.get(j).predict(picked_instance) == corresponding_label) ? 1.0 : 0.0);
				}
				
				// return the picked instance to the original dataset
				dataset.add_datapoint(picked_instance, corresponding_label);
			}
		}
		else {
			if (dataset.size() % k != 0)
				System.out.println("Warning size of dataset is not a multiple of k, leaving " + dataset.size() % k + " instances\n");
			
			int group_size = dataset.size() / k; // fold size
			
			List<Integer> all_indices = IntStream.range(0, dataset.size()).boxed().collect(Collectors.toList());
			// shuffle the indices list
			Collections.shuffle(all_indices);
			
			// constructing a list of indices
			List<List<Integer>> k_index_lists = new ArrayList<>();
			
			for (int i = 0; i < k; i++) {
				// get the [group_size] first indices from all_lists list	
				k_index_lists.add(all_indices.stream().limit(group_size).collect(Collectors.toList()));
				
				// remove them so we don't get the same indices the next iteration
				all_indices.removeAll(all_indices.stream().limit(group_size).collect(Collectors.toList()));
			}
			
			// perform the k-fold cross val
			for (int test_index = 0; test_index < k; test_index++) {
				// constructing the kth_test_dataset
				Dataset test_dataset = new Dataset(); // very interesting behaviour, declaring a var inside the loop solves the lambdas final/eventually_final field issue.
				k_index_lists.get(test_index).stream().map(datapoints::get).forEach(img -> test_dataset.add_datapoint(img, img.get_label()));
				
				// the rest is for training
				Dataset train_dataset = new Dataset();
				int test_index_copy = test_index; // just to surpass the eventually final constraint
				List<Integer> training_indices = k_index_lists.stream()
															  .flatMap(List::stream)
															  .filter(i -> !k_index_lists.get(test_index_copy).contains(i))
															  .collect(Collectors.toList());
				
				training_indices.stream().map(datapoints::get).forEach(img -> train_dataset.add_datapoint(img, img.get_label()));
				
				for (int j = 0; j < classifiers.size(); j++) {
					// training each classifier on new dataset ( the other K-1 groups/folds ) 
					classifiers.get(j).train(train_dataset);
					
					// adding results to each list
					training_accs.get(j).add(classifiers.get(j).accuracy(train_dataset));
					test_accs.get(j).add(classifiers.get(j).accuracy(test_dataset));
					
					// reset the classifiers to have a "fresh" ones in the next iteration
					// nearly useless with our two classifiers for now, but may come in handy in the future
					classifiers.get(j).reset();
				}
				
			}
		}
		
		// construct the results to be returned
		for (int j = 0; j < classifiers.size(); j++) {
			
			double avg_training_accuracy = training_accs.get(j).stream().mapToDouble(v -> v).average().orElse(0.0);
			double avg_test_accuracy = test_accs.get(j).stream().mapToDouble(v -> v).average().orElse(0.0);
			
			double training_std_dev = Math.sqrt(training_accs.get(j).stream()
																	.mapToDouble(v -> Math.pow((v - avg_training_accuracy), 2))
																	.sum()) / training_accs.get(j).size();
			
			double test_std_dev = Math.sqrt(test_accs.get(j).stream()
															.mapToDouble(v -> Math.pow((v - avg_test_accuracy), 2))
															.sum()) / test_accs.get(j).size();
			
			
			results.add(new CrossValEntry(avg_training_accuracy, training_std_dev, avg_test_accuracy, test_std_dev));
		}
		
		if (verbose) 
			for (int j = 0; j < classifiers.size(); j++) {
				System.out.println(classifiers.get(j) + " stats: ");
				System.out.println("-> List of training accuracies: " + training_accs.get(j));
				System.out.println("Average = " + results.get(j).get_training_acc() + ", Stddev = " + results.get(j).get_training_std());
				System.out.println("-> List of test accuracies: " + test_accs.get(j));
				System.out.println("Average = " + results.get(j).get_test_acc() + ", Stddev = " + results.get(j).get_test_std() + "\n");
			}
		
		return results;
	}
	
	public static List<CrossValEntry> cross_validation(List<AbstractClassifier> classifiers, Dataset dataset, int k) {
		return cross_validation(classifiers, dataset, k, false, true);
	}
	
	public static List<ConfusionMatrix> confusion_matrix(List<AbstractClassifier> classifiers, Dataset train_dataset, Dataset test_dataset) {
		assert classifiers.size() > 0;
		
		List<ConfusionMatrix> matrices = new ArrayList<>(); // to be returned
		Set<Image> datapoints = test_dataset.keySet();
		
		// get the true labels respectively
		List<Integer> true_labels = datapoints.stream().map(img -> img.get_label()).collect(Collectors.toList()); 
		
		for (AbstractClassifier classifier : classifiers) {
			// train the classifer on the training dataset
			classifier.train(train_dataset);
			
			// construct the predicted labels list
			List<Integer> predicted_labels = datapoints.stream().map(img -> classifier.predict(img)).collect(Collectors.toList());
			
			// construct the confusion matrix of the classifer and add it to the list
			matrices.add(new ConfusionMatrix(true_labels, predicted_labels));
		}
		
		return matrices;
	}

	// special case: just one classifier
	@SuppressWarnings("serial")
	public static CrossValEntry cross_validation(AbstractClassifier classifier, Dataset dataset, int k, boolean verbose, boolean shuffle) {
		return cross_validation(new ArrayList<AbstractClassifier>(){{add(classifier);}}, dataset, k, verbose, shuffle).get(0);
	}
	
	@SuppressWarnings("serial")
	public static CrossValEntry cross_validation(AbstractClassifier classifier, Dataset dataset, int k) {
		return cross_validation(new ArrayList<AbstractClassifier>(){{add(classifier);}}, dataset, k, false, true).get(0);
	}
	
	@SuppressWarnings("serial")
	public static ConfusionMatrix confusion_matrix(AbstractClassifier classifier, Dataset train_dataset, Dataset test_dataset) {

		return confusion_matrix(new ArrayList<AbstractClassifier>(){{add(classifier);}}, train_dataset, test_dataset).get(0);
	}
	
}

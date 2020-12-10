package eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
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
	 * Comparing a list of classifiers accuracy ratio given train and test datasets.
	 */
	public static Map<AbstractClassifier, Double> accuracy(List<AbstractClassifier> classifiers, Dataset train_set, Dataset test_set, boolean verbose) {
		/*
		 * This method will train all classifiers on train_set, and calculate the accuracy on both train and test sets.
		 * 
		 * Only accuracies of test set will be returned in the form of a map < Classifier, his_test_accuracy >
		 * 
		 * Verbose will print the results, for each classifier his training and test accuracies.
		 * 
		 */
		
		assert classifiers.size() > 0;
		for (int i = 0; i < classifiers.size(); i++)
			assert classifiers.get(i) != null;
		
		Map<AbstractClassifier, Double> accuracies_map = new LinkedHashMap<>();
		
		// train all classifiers
		for (AbstractClassifier classifier : classifiers)
			classifier.train(train_set);
		
		// calculate the accuracies and print infos if verbose
		double test_acc;
		for (AbstractClassifier classifier : classifiers) {
			test_acc = classifier.accuracy(test_set);
			accuracies_map.put(classifier, test_acc);
			
			if(verbose)
				System.out.println(classifier + " -> train_acc : " + classifier.accuracy(train_set) + " ; test_acc : " + test_acc + "\n");
		}
		
		return accuracies_map;
	}
	
	/*
	 * Comparing a list of classifiers with K-fold cross validation.
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
			if (verbose)
				System.out.println("Performing leave one out..." + '\n');
			
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
		for (int i = 0; i < classifiers.size(); i++)
			assert classifiers.get(i) != null;
		
		List<ConfusionMatrix> matrices = new ArrayList<>(); // to be returned
		Set<Image> datapoints = test_dataset.keySet();
		
		// get the true labels respectively
		List<Integer> true_labels = datapoints.stream().map(img -> img.get_label()).collect(Collectors.toList()); 
		
		for (AbstractClassifier classifier : classifiers) {
			// train the classifer on the training dataset
			classifier.train(train_dataset);
			System.out.println(classifier.accuracy(train_dataset));
			
			// construct the predicted labels list
			List<Integer> predicted_labels = datapoints.stream().map(img -> classifier.predict(img)).collect(Collectors.toList());
			
			// construct the confusion matrix of the classifer and add it to the list
			matrices.add(new ConfusionMatrix(true_labels, predicted_labels));
		}
		
		return matrices;
	}
	
	public static List<Double> train_test_split_accuracy(List<AbstractClassifier> classifiers, Dataset dataset, double train_percentage, boolean verbose) {
		/*
		 * This method will split the dataset into two datasets, using the training_percentage.
		 * Then it will train the passed classifiers on the training dataset, and then test them on the test_set
		 * 
		 * returns: a list of the corresponding test accuracies
		 */
		assert classifiers.size() > 0;
		for (int i = 0; i < classifiers.size(); i++)
			assert classifiers.get(i) != null;
		
		// split the dataset
		List<Dataset> d = split_dataset(dataset, train_percentage);
		Dataset train_set= d.get(0);
		Dataset test_set= d.get(1);
		
		if (verbose) 
			System.out.println("Training dataset size = " + train_set.size() + " while test set size is " + test_set.size() + "\n");
		
		
		List<Double> test_accuracies = new ArrayList<>();
		// train all classifiers
		for (AbstractClassifier classifier : classifiers) {
			classifier.train(train_set);
			
			double test_acc = classifier.accuracy(test_set);
			
			if (verbose) 
				System.out.println(classifier + " had a training acc of : " + classifier.accuracy(train_set) + " and test acc: " + test_acc);
			
			test_accuracies.add(test_acc);
		}
		
		if (verbose) {
			int best_cl_index = test_accuracies.indexOf(Collections.max(test_accuracies));
			System.out.println("The best classifier is " + classifiers.get(best_cl_index) + " with train acc = " + classifiers.get(best_cl_index).accuracy(train_set) + " with a test acc of " + test_accuracies.get(best_cl_index));
		}
		
		return test_accuracies;
	}

	@SuppressWarnings("serial")
	public static List<Dataset> split_dataset(Dataset dataset, double train_percentage) {
		/*
		 *  will do stratified random sampling by taking split_percentage of each class.
		 *  
		 *  Returns a list of datasets, first is training set second is test set. 
		 */
		
		assert dataset != null && dataset.size() > 0;
		assert train_percentage > 0 && train_percentage < 1;
		
		Dataset train_set = new Dataset();
		Dataset test_set = new Dataset();
		
		int test_sample_size, label;
		// for each class/stratum get random sample of size stratum_size * split_percentage, the rest to the test_set
		for (Map.Entry<Integer, List<Image>> entry : dataset.get_stratums().entrySet()) {
			// construct a list of indices of the size of the current stratum
			List<Integer> indices = IntStream.range(0, entry.getValue().size()).boxed().collect(Collectors.toList());
			
			// shuffle the list
			Collections.shuffle(indices);
			
			// take the first (stratum_size -  [stratum_size * split_percentage]) and add them to the test_set
			test_sample_size = indices.size() - (int) Math.floor(indices.size() * train_percentage);
			label = entry.getKey();
			
			for (int i = 0; i < test_sample_size; i++) {
				Image img = entry.getValue().get(indices.get(i));
				test_set.add_datapoint(img, label);
			}
			
			// rest to train set
			for (; test_sample_size < indices.size(); test_sample_size++) {
				Image img = entry.getValue().get(indices.get(test_sample_size));
				train_set.add_datapoint(img, label);
			}
			
		}
		
//		System.out.println(train_set.size() + " + " + test_set.size() + " = " + dataset.size());
//		
//		List<Image> imgs = test_set.keySet().stream().collect(Collectors.toList());
//		imgs.addAll(train_set.keySet());
//		
//		System.out.println(dataset.keySet().stream().collect(Collectors.toList()).containsAll(imgs));
//		System.out.println(imgs.containsAll(dataset.keySet().stream().collect(Collectors.toList())));
		
//		System.out.println(train_set.get_stratums().keySet().equals(test_set.get_stratums().keySet()));
		
		return new ArrayList<Dataset>() {{
			add(train_set);
			add(test_set);
		}};
	}
	
	// special cases: just one classifier
	public static Map<AbstractClassifier, Double> accuracy(List<AbstractClassifier> classifiers, Dataset train_set, Dataset test_set) {
		return accuracy(classifiers, train_set, test_set, true);
	}
	
	@SuppressWarnings("serial")
	public static Map<AbstractClassifier, Double> accuracy(AbstractClassifier classifier, Dataset train_set, Dataset test_set) {
		return accuracy(new ArrayList<AbstractClassifier>(){{add(classifier);}}, train_set, test_set, true);
	}
	
	@SuppressWarnings("serial")
	public static Map<AbstractClassifier, Double> accuracy(AbstractClassifier classifier, Dataset train_set, Dataset test_set, boolean verbose) {
		return accuracy(new ArrayList<AbstractClassifier>(){{add(classifier);}}, train_set, test_set, verbose);
	}
	
	
	@SuppressWarnings("serial")
	public static CrossValEntry cross_validation(AbstractClassifier classifier, Dataset dataset, int k) {
		return cross_validation(new ArrayList<AbstractClassifier>(){{add(classifier);}}, dataset, k, false, true).get(0);
	}
	
	@SuppressWarnings("serial")
	public static CrossValEntry cross_validation(AbstractClassifier classifier, Dataset dataset, int k, boolean verbose) {
		return cross_validation(new ArrayList<AbstractClassifier>(){{add(classifier);}}, dataset, k, verbose, true).get(0);
	}
	
	@SuppressWarnings("serial")
	public static CrossValEntry cross_validation(AbstractClassifier classifier, Dataset dataset, int k, boolean verbose, boolean shuffle) {
		return cross_validation(new ArrayList<AbstractClassifier>(){{add(classifier);}}, dataset, k, verbose, shuffle).get(0);
	}
	
	@SuppressWarnings("serial")
	public static ConfusionMatrix confusion_matrix(AbstractClassifier classifier, Dataset train_dataset, Dataset test_dataset) {

		return confusion_matrix(new ArrayList<AbstractClassifier>(){{add(classifier);}}, train_dataset, test_dataset).get(0);
	}
	
}

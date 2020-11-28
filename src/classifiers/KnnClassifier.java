package classifiers;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;

import dataset.Dataset;
import dataset.Image;

import java.util.TreeMap;

public class KnnClassifier extends AbstractClassifier{
	/*
	 * Knn classifier
	 * @params : k (int)
	 *           p (int) which distance that will be used (p param in the Minkowski distance)
	 *           
	 */
	private int k, p; // p for which distance
	private Dataset training_dataset;
	
	public KnnClassifier(int k) { // if p is not passed -> use euclidean distance. 
		this.k = k;
		this.p = 2;
	}
	
	public KnnClassifier(int k, int p) {
		this(k);
		this.p = p;
	}

	@Override
	public boolean train(Dataset training_dataset) {
		assert training_dataset.size() > 0;
		this.training_dataset = training_dataset;
		return true;
	}

	@Override
	public int predict(Image img) {
		/*
		 * Key = distance between the passed img and an image from the training dataset
		 * Value = the label of the corresponding training img
		 * Using TreeMap (Red-Black tree) to keep the map sorted by Key
		 * hence each inserted distance will be sorted automatically.
		 * insertion complexity is O(log n)
		 * 
		 * Getting the first element is done in O(1), thus avoiding the sorting...
		 */
		  
		Map<Double, Integer> dist_array = new TreeMap<>();
		
		// calculating all distances between the passed img and images in training_dataset
		// and storing the pair <Distance, Label> in the Treemap
		for(Entry<Image, Integer> training_entry : this.training_dataset.entrySet()) 
			dist_array.put( dist(img, training_entry.getKey(), this.p), training_entry.getValue() );
		
		// getting the first kth labels from the treeMap
		List<Integer> labels = new ArrayList<>();
		int count = 0;
		
		for(Map.Entry<Double,Integer> entry : dist_array.entrySet()) {
			if(count++ == this.k)
				break;
			
			labels.add(entry.getValue());
		}
		
		// now labels is a list that contains the labels of the nearest kth neighbors
		// returning the most frequent label, i.e : calculating the mode
		int mode = labels.stream()
						 .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
						 .entrySet()
						 .stream()
						 .max(Comparator.comparing(Entry::getValue))
						 .get().getKey();
		
		return mode;
	}
	

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("KNN Classifier with k = " + k + " and p = " + p);
		
		
		return sb.toString();
	}

}

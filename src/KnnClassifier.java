import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
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
		this.training_dataset = training_dataset;
		return true;
	}

	@Override
	public double score(Image img) {
		// USELESS IN OUR MULTICLASS KNN CLASSIFIER
		return 0;
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
	
	
	public double dist(double[] x, double[] y, int p) {
		// generally for debugging purposes
		assert p == 1 || p == 2;
		
		if (p == 1)
			return manhattan_dist(x, y);
		
		return euclidean_dist(x, y);
	}
	
	// Overloading of the method above to make things simpler
	
	public double dist(List<Double> l_x, List<Double> l_y, int p) {
		// generally for debugging purposes
		assert p == 1 || p == 2;
		
		double[] x = convert_list_to_double(l_x);
		double[] y = convert_list_to_double(l_y);
		
		if (p == 1)
			return manhattan_dist(x, y);
		
		return euclidean_dist(x, y);
	}
	
	public double dist(Representation r_x, Representation r_y, int p) {	
		// will be used in production
		return dist(r_x.get_data(), r_y.get_data(), p);
	}
	
	public double dist(Image img_x, Image img_y, int p) {
		// will be used in production
		return dist(img_x.get_representation(), img_y.get_representation(), p);
	}
	
	private double euclidean_dist(double[] x, double[] y) {
		assert x.length == y.length;
		
		double sum = 0;
		
		for (int i = 0; i < x.length; i++)
			sum += Math.pow((x[i] - y[i]), 2);
		
		return Math.sqrt(sum);
	}
	
	private double manhattan_dist(double[] x , double[] y) {
		assert x.length == y.length;
		
		double sum = 0;
		
		for (int i = 0; i < x.length; i++)
			sum += Math.abs((x[i] - y[i]));
		
		return sum;
	}
	
	private double[] convert_list_to_double(List<Double> l) {
		boolean null_exists = l.stream().anyMatch(d -> d == null);
		
		if (null_exists) 
			System.err.println("Warning: Null exists in the passed list, will be replaced by 0...");
		
		double[] arr = l.stream().map(i -> (i == null ? 0 : i))
								 .mapToDouble(Double::doubleValue)
								 .toArray();
		
		return arr;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("KNN Classifier with k = " + k + " and p = " + p);
		
		
		return sb.toString();
	}

}

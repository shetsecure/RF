package classifiers;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import dataset.Dataset;
import dataset.Image;

public class KmeansClassifier extends AbstractClassifier {
	/*
	 * Two initializations method:
	 * 
	 * 	   1) Choosing the first K-centroids is done randomly in K-means. Doing it completely random will result
	 * 	   in a very slow convergence speed ( so more iterations are needed ).
	 * 	   To overcome this problem, we need to determine the min-max range for each attribute. And in order to do so,
	 * 	   we need to iterate over the all the attribute's values over all the instances of the dataset.
	 * 	   To avoid this, I kept two lists of mins and maxs in Dataset that gets updated in each insertion.
	 *     
	 * 
	 *          
	 *     2) K-means++ chooses its initial centroids using D^2 weighting.
	 *     Which can add some cost in the initialization step, but make it faster in the end ( less iterations )
	 * 	   K-means++ is faster and more accurate
	 * 	   More in: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
	 * 
	 */
	
	private int k, p; // number of clusters, and p-> p param of Minkowski metric
	private long max_iter;
	private String init_method; // which init method will be used
	
	private Dataset training_dataset;
	private Map<Centroid, List<Image>> clusters; // list of constructed clusters
	
	private static final Random random = new Random();
	
	@SuppressWarnings("unused")
	private KmeansClassifier() {} // forcing the user to specify k and p
	
	public KmeansClassifier(int k, int p) {
		this(k, p, 100, "kmeans++"); // use k-means++ by default
	}
	
	public KmeansClassifier(int k, int p, long max_iter) {
		this(k, p, max_iter, "kmeans++");
	}
	
	public KmeansClassifier(int k, int p, long max_iter, String init_method) {
		assert k > 0;
		assert p > 0;
		assert max_iter > 0;
		assert init_method.toLowerCase().equals("random") // completely random ( i.e random from min_value to max_value )
			|| init_method.toLowerCase().equals("enhanced_random") // enhanced random takes random from specific range, see below
			|| init_method.toLowerCase().equals("kmeans++"); // kmeans++ init
		
		
		this.k = k;
		this.p = p;
		this.max_iter = max_iter;
		this.init_method = init_method.toLowerCase();
		
		this.clusters = new LinkedHashMap<>();
	}

	@Override
	public boolean train(Dataset training_dataset) {
		assert training_dataset != null && training_dataset.size() > 0;
		
		this.training_dataset = training_dataset;
		boolean enhanced = false;
		
		// STEP 1: INITIALIZATION
		// 1.a Get the initial_centroids
		List<Centroid> initial_centroids;
		
		if (init_method.equals("random"))
			initial_centroids = get_random_centroids(false);
		else if (init_method.equals("enhanced_random")) {
			initial_centroids = get_random_centroids(true);
			enhanced = true;
		}
		else
			initial_centroids = kmeans_plus_plus_init();
				
		// STEP 2
		
		/*
		 * At first avg_centroids are the initial centroids, and we need to create another list that keeps track of 
		 * the last avg_centroids, so when nothing changes, we stop. 
		 *  Or we stop if we reach max_iter
		 */
		
		List<Centroid> last_centroids;
		List<Centroid> avg_centroids = initial_centroids.stream().map(Centroid::clone).collect(Collectors.toList());
		
		long counter = 0; // useful to see if we reached max_iter
		
		do {
			// get a deep copy // copy the last_avg_computed_centroids to check later if nothing changed
			last_centroids = avg_centroids.stream().map(Centroid::clone).collect(Collectors.toList());
			
			// constructing the new clusters
			clusters = new LinkedHashMap<Centroid, List<Image>>();
			for (Centroid centroid : last_centroids)
				clusters.put(centroid, new ArrayList<>());
			
			// STEP 2: For each i∈{1, . . . , k}, set the cluster C_i to be the set of points in X that are closer to
			// c_i than they are to c_j for all j != i
			// Iter over the dataset, and assign each image to its nearest centroid
			for (Image img : training_dataset.keySet()) {
				
				Centroid nearest_centroid = get_the_nearest_centroid(img, last_centroids).getKey();

				clusters.get(nearest_centroid).add(img);
			}
			
			// Calculate the new avg centroids
			avg_centroids = new ArrayList<>();
			
			// STEP 3: For each i∈{1, . . . , k}, set ci to be the center of mass of all points in Ci : ci= 1/Ci * ∑_x∈Ci x
			for (Centroid centroid : last_centroids) {
				List<Image> imgs = clusters.get(centroid);
				avg_centroids.add(get_avg_centroid(centroid, imgs));
			}
			
		} while (! last_centroids.equals(avg_centroids) && counter++ < max_iter);
		
		System.out.println("iterations = " + counter);
		
		List<Integer> possible_labels = IntStream.range(1, 10).boxed().collect(Collectors.toList());
		
		// fix the label of the final centroids as the dominant label
		for (Centroid centroid : clusters.keySet()) {
			int label = get_dominant_label(clusters.get(centroid));
			possible_labels.removeIf(l -> possible_labels.contains(l) && l == label);
			
			centroid.set_label(label);
		}
		
		if (enhanced) {
			// Handling the case of empty clusters if we did a random initialization
			// randomly distribute non-used labels to isolated centroids
			for (Centroid centroid : clusters.keySet()) {
				if (centroid.get_label() == -100) {
					int random_index = random.nextInt(possible_labels.size());
					centroid.set_label(possible_labels.get(random_index));
					
					possible_labels.remove(random_index);
				}
			}
		}
		return true;
	}

	@Override
	public int predict(Image img) {
		assert img.get_representation_type().equals(training_dataset.get_representation_type());
		
		return get_the_nearest_centroid(img, new ArrayList<>(clusters.keySet()) ).getKey().get_label();
	}
	
	@Override
	public void reset() {
		this.clusters = new LinkedHashMap<>();
	}
	
	public Map<Centroid, List<Image>> get_clusters() {
		return clusters;
	}

	private List<Centroid> get_random_centroids(boolean enhanced) {
		/*
		 * enhanced random will take random value in [min, max] range for each attribute
		 * 
		 * completely random will take random value in [MIN_VALUE, MAX_VALUE]
		 */
		List<Centroid> random_centroids = new ArrayList<>();
		
		for(int i = 0; i < this.k; i++) 
			random_centroids.add(new Centroid(get_random_coords(enhanced)));
		
		return random_centroids;
	}
	
	private List<Double> get_random_coords(boolean enhanced) {
		/*
		 * enhanced random will take random value in [min, max] range for each attribute
		 * 
		 * completely random will take random value in [MIN_VALUE, MAX_VALUE]
		 */
		
		List<Double> random_coords = new ArrayList<>();
		List<Double> mins = training_dataset.get_mins();
		List<Double> maxs = training_dataset.get_maxs();
		
		int dim = mins.size();
		double max, min;
		
		if (enhanced) {
			// getting the range of [min-max] list for each attribute 
			
			for(int i = 0; i < dim; i++) {
				max = maxs.get(i);
				min = mins.get(i);
				
				random_coords.add(random.nextDouble() * (max - min) + min);
			}
		}
		else {
			for(int i = 0; i < dim; i++) {
				max = Double.MAX_VALUE;
				min = Double.MIN_VALUE;
				
				random_coords.add(random.nextDouble() * (max - min) + min);
			}
		}
		
		return random_coords;
	}
	
	private List<Centroid> kmeans_plus_plus_init() {
		// using Kmeans++
		
		// 2.2 in http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
		// initialization phase using Kmeans++ approach: D^2 weighting
		// constructing the array of centroids
		List<Centroid> centroids = new ArrayList<>();
		
		// getting the list of all datapoints
		List<Image> all_datapoints = new ArrayList<>(training_dataset.keySet());
		
//		assert k <= all_datapoints.size(); 
		
		// Take one center c0, chosen uniformly at random from X (all_datapoints)
		Image selected_img = all_datapoints.get(random.nextInt(all_datapoints.size()));
//		Image selected_img = all_datapoints.get(7);
		centroids.add(new Centroid(selected_img.get_representation().get_data()));
		
		// remove the selected_img from all_datapoints list so we don't calculate its
		// distance to all the centroids ( because it's already a centroid)
		// https://www.youtube.com/watch?v=HatwtJSsj5Q
		all_datapoints.remove(selected_img);
		
		// 1c. Repeat Step 1b. until we have taken k centers altogether
		for (int i = 1; i < this.k; i++) {
			// 1b. Take a new center ci, choosing x∈X with probability D(x)^2 / (sum[for x in X] of D(x)^2)
			
			// calculating all distances from all datapoints to all current centroidS
			
			double sum = 0; // (sum[for x in X] of D(x)^2)
			List<Double> distances = new ArrayList<>(); // each entry will hold the MIN distance between all centroids and datapoint_i;
			
			// filling distances list i.e calculating D(x)^2 for x∈X  
			for (Image img : all_datapoints) {
				double dist = get_the_nearest_centroid(img, centroids).getValue();
				distances.add(dist);
				sum += dist;
			}
			
			// calculating the proportions/probabilities list, all values now are between 0 and 1
			List<Double> probabilities = new ArrayList<>();
			
			for (Double distance : distances)
				probabilities.add(distance / sum); // probability = D(x)^2 / (sum[for x in X] of D(x)^2)
			
			// Weighted random choice using the probabilities list ( get the index )
			int chosen_centroid_index = weighted_random(probabilities);
//			int chosen_centroid_index = probabilities.indexOf(Collections.max(probabilities));
			
			// choosing x∈X with probability D(x)^2 / (sum[for x in X] of D(x)^2)
			// add the chosen img[i] to centroids list and remove it from datapoints list
			selected_img = all_datapoints.get(chosen_centroid_index); 
			centroids.add(new Centroid(selected_img.get_representation().get_data()));
			all_datapoints.remove(selected_img);
		}
		
		return centroids;
	}
	
	private Centroid get_avg_centroid(Centroid centroid, List<Image> imgs) {
		if (imgs == null || imgs.size() == 0) 
			return centroid;
		
		List<Double> avg_coords = new ArrayList<>();
		
		int h_many_imgs = imgs.size(); // how many images in the dataset
		int dim = training_dataset.get_maxs().size(); // how many attributes in each image
		
		double sum;
		
		for (int i = 0; i < dim; i++) {
			sum = 0;
			
			for (Image img : imgs)
				sum += img.get_representation().get_data().get(i);
			
			avg_coords.add(sum / h_many_imgs);
		}
		
		return new Centroid(avg_coords);
	}
	
	private Map.Entry<Centroid, Double> get_the_nearest_centroid(Image img, List<Centroid> centroids) {
		/*
		 * will calculate all distances between the image img, and all centroids in
		 * the list then returns the pair <Nearest_centroid, the minimum distance>.
		 * no Pair class in java -> using Map.Entry
		 */
		
		assert centroids != null && centroids.size() > 0;
		
		List<Double> img_coords = img.get_representation().get_data();
		
		int min_index = 0;
		double min_distance = dist(img_coords, centroids.get(0).get_coords(), this.p);
		double new_distance;
		
		for (int i = 1; i < centroids.size(); i++) {
			new_distance = dist(img_coords, centroids.get(i).get_coords(), this.p);
			
			if (new_distance < min_distance) {
				min_distance = new_distance;
				min_index = i;
			}
		}
		
		return new AbstractMap.SimpleImmutableEntry<>(centroids.get(min_index), min_distance);
	}
	
	private int get_dominant_label(List<Image> images) {
		if (images == null || images.size() == 0)
			return -100;
		
		List<Integer> labels = new ArrayList<>();
		
		for (Image img : images)
			labels.add(img.get_label());
		
		int mode = labels.stream()
						 .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
						 .entrySet()
						 .stream()
						 .max(Comparator.comparing(Entry::getValue))
						 .get().getKey();

		return mode;
	}
	
	private int weighted_random(List<Double> proportions) {
		/*
		 * Randomly select an element e according to the probability distribution that was passed (proportions).
		 * here our elements are the indices.
		 * 
		 * Example:
		 * 		elements = [0, 1, 2, 3]
		 * 		Random.nextInt(elements.size() = 4) will choose a uniformly distributed number between 0 and 3
		 * 		that is each element has a probability of 1/4 to get selected
		 * 
		 * What we want to do is:
		 * 		given elements = [0, 1, 2, 3], and proportions [p0, p1, p2, p3] (N.B sum of proportions = 1)
		 * 		So each proportion p_i can be written as: a_i/ B ( where sum of all a_i = B )
		 * 		choose an element according to the proportions that were passed.
		 * 		
		 * 		We can do this, by constructing an array A with a length = B, and adding each element e_i to 
		 * 		A a_i times, then pick uniformly an element from that array.
		 * 		The Problem of this is it has a space complexity of O(B).
		 * 
		 * 		Hence the alternative approach below.
		 * 
		 * Having the proportions list, we can pick a uniformly distributed random number between 0 and 1 r_p
		 * then as we iterate over the proportions list, we can calculate the cummulative probability C_p.
		 * Once C_p becomes bigger than our r_p, then we return the element at that specific index.
		 * 
		 * The order of the items in the list doesn't matter, because the value of r_p is a uniformly distributed
		 * random number.Thus the items in the list are not favored and it doesn't matter where they are in the list.
		 * So no sorting is needed.
		 * 
		 */
		assert proportions.size() > 0;
		
		double r_p = random.nextDouble();
		double cum_proba = 0.0; // cummulative probability
		
		for (int i = 0; i < proportions.size(); i++) {
			cum_proba += proportions.get(i);
			
			if (cum_proba >= r_p )
				return i;
		}
		
		return 0;
	}
 	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("Kmeans classifier with init_method = " + init_method + ", k = " + k + " p = " + p + "\n");
//		sb.append("Constructed centroids are : " + "\n" + clusters.keySet());
		
		return sb.toString();
	}
}

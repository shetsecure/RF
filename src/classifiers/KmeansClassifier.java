package classifiers;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import dataset.Dataset;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import dataset.*;

public class KmeansClassifier extends AbstractClassifier {
	/*
	 * We'll use K-means++ algorithm instead of the standard K-means one for two reasons:
	 * 
	 * 	1) Choosing the first K-centroids is done randomly in K-means. Doing it completely random will result
	 * 	   in a very slow convergence speed ( so more iterations are needed ).
	 * 	   To overcome this problem, we need to determine the min-max range for each attribute. And in order to do so,
	 * 	   we need to iterate over the all the attribute's values over all the instances of the dataset.
	 * 
	 * 	   Assuming that each instance/example/image in our dataset is a d-dimensional vector, with S = size of our dataset
	 * 	   then we'll have to determine the min and the max of vector that has a dimension of (1, d*S).
	 * 	   Time complexity will be O(d*S)
	 * 
	 *          
	 *     Instead, K-means++ chooses its initial centroids using D^2 weighting in O(log(K))
	 *     Which can add some cost in the initialization step, but make it faster in the end ( less iterations )
	 *     
	 * 	2) K-means++ is faster and more accurate
	 * 	   More in: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
	 * 
	 */
	
	private int k, p; // number of clusters, and p-> p param of Minkowski metric
	private long max_iter; 
	private Dataset training_dataset;
	public Map<Image, Integer> centroids_map; // list of constructed centroids and their corresponding label
	private static final Random random = new Random();
	 
	public KmeansClassifier() {
		// 9 number of clusters, p = 2 for using euc distance
		// 100 for max_iter
		this(9, 2, 100);
	}
	
	public KmeansClassifier(int k) {
		this(k, 2, 100); // 2 and 100 are p_norm and max_iter respectively.
	}
	
	public KmeansClassifier(int k, int p) {
		this(k, p, 100); // 100 is max_iter.
	}
	
	public KmeansClassifier(int k, int p, long max_iter) {
		assert k >= 1 && k <= 9;
		assert p == 1 || p == 2;
		assert max_iter > 0;
		
		this.k = k;
		this.p = p;
		this.max_iter = max_iter;
		this.centroids_map = new HashMap<Image, Integer>();
	}

	@Override
	public boolean train(Dataset training_dataset) {
		assert training_dataset.size() > 0;
		
		this.training_dataset = training_dataset;
		
		// place the k centroids randomly
		List<Image> initial_centroids = init();
		
		if(initial_centroids == null || initial_centroids.size() < 1) {
			System.err.println("Something bad happend. Couldn't generate initial centroids");
			return false;
		}
		
		/*
		 * At first avg_centroids are the initial centroids, and we need to create another list that keeps track of 
		 * the last avg_centroids, so when nothing changes, we stop. 
		 *  Or we stop if we reach max_iter
		 */
		List<Image> avg_centroids = initial_centroids.stream().map(Image::clone)
													 .collect(Collectors.toList());
		
		// deep copy the list of last avg_centroids, will serve to check if we changed something or not
		List<Image> last_centroids; 
		
		long counter = 0; // useful to see if we reached max_iter
		
		// I put these two here to augment their scope to use them later
		Map<Image, List<Image>> current_clusters, avg_clusters;
		
		do {
			// copy the last_avg_computed_centroids to check later if nothing changed
			last_centroids = avg_centroids.stream().map(Image::clone)
												   .collect(Collectors.toList());
		
			// STEP 2: For each i∈{1, . . . , k}, set the cluster C_i to be the set of points in X that are closer to
			// c_i than they are to c_j for all j != i
			current_clusters = construct_new_clusters(last_centroids);
			
			// STEP 3: For each i∈{1, . . . , k}, set ci to be the center of mass of all points in Ci : ci= 1/Ci * ∑_x∈Ci x
			avg_clusters = get_avg_clusters(current_clusters);
			
			// construct the list of the new avg_centroids
			avg_centroids = avg_clusters.keySet().stream().collect(Collectors.toList());
			
		} while(!avg_centroids.equals(last_centroids) && counter++ < max_iter);
		
		System.out.println("iterations = " + counter);
		
		// create the final centroids map
		for( Map.Entry<Image, List<Image>> cluster : avg_clusters.entrySet()) 
			this.centroids_map.put(cluster.getKey(), cluster.getKey().get_label());
		
		return true;
	}

	@Override
	public int predict(Image img) {
		Image nearest_centroid_to_img = get_the_nearest_centroid(img, new ArrayList<>(centroids_map.keySet())).getKey();
		return centroids_map.get(nearest_centroid_to_img);
	}

	@Override
	public String toString() {
		return "K-means Classifier with K = " + k + " and max_iter = " + max_iter;
	}
	
	private List<Image> init() {
		// 2.2 in http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
		// initialization phase using Kmeans++ approach: D^2 weighting
		// constructing the array of centroids
		List<Image> centroids = new ArrayList<>();
		
		// getting the list of all datapoints
		List<Image> all_datapoints = new ArrayList<>(training_dataset.keySet());
		
		assert k <= all_datapoints.size(); 
		
		// Take one center c0, chosen uniformly at random from X (all_datapoints)
//		Image selected_img = all_datapoints.get(random.nextInt(all_datapoints.size()));
		Image selected_img = all_datapoints.get(7);
		centroids.add(selected_img);
		
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
			
			// choosing x∈X with probability D(x)^2 / (sum[for x in X] of D(x)^2)
			// add the chosen img[i] to centroids list and remove it from datapoints list
			selected_img = all_datapoints.get(chosen_centroid_index); 
			centroids.add(selected_img);
			all_datapoints.remove(selected_img);
		}
		
		return centroids;
	}
	
	// construct_clusters(last_centroids, training_dataset) -> Map<Image, List<Image>>
	private Map<Image, List<Image>> construct_new_clusters(List<Image> last_centroids) {
		// For each i∈{1, . . . , k}, set the cluster C_i to be the set of points in X that are closer to
		// c_i than they are to c_j for all j != i
		
		// the map which will hold the clusters, key = centroid, value = assigned images
		Map<Image, List<Image>> clusters = new HashMap<>();
		
		// adding the last_centroids as the keys, and init the values as an empty list
		for (Image centroid : last_centroids) 
			clusters.put(centroid, new ArrayList<Image>());
		
		// for each datapoint (image), assign it to its nearest centroid
		for (Image image : training_dataset.keySet()) {
			Image nearest_centroid = get_the_nearest_centroid(image, last_centroids).getKey();
			
			// assign the current image to its nearest centroid
			clusters.get(nearest_centroid).add(image);
		}
		
		
//		for (Image centroid : last_centroids)
//			if(clusters.get(centroid).size() == 0) {
//				System.out.println(centroid.get_representation().get_data().size());
//			}
		return clusters;
	}
	
	private Map.Entry<Image, Double> get_the_nearest_centroid(Image img, List<Image> centroids) {
		// will calculate all distances between the image img, and all centroids in the list 
		// then returns the pair <Nearest_centroid, the minimum distance>. 
		// no Pair class in java -> using Map.Entry
		
		assert centroids.size() > 0;
		
		double min_dist = squared_dist(img, centroids.get(0));
		double curr_dist;
		int min_index = 0;
		
		for (int i = 1; i < centroids.size(); i++) {
			curr_dist = squared_dist(img, centroids.get(i));
			if ( curr_dist < min_dist) {
				min_dist = curr_dist;
				min_index = i;
			}
		}
		
		return new AbstractMap.SimpleImmutableEntry<>(centroids.get(min_index), min_dist);
	}
	
	// get_avg_clusters(Map<Image, List<Image>>) -> Map<Image, List<Image>>
	private Map<Image, List<Image>> get_avg_clusters(Map<Image, List<Image>> current_clusters) {
		// For each i∈{1, . . . , k}, set ci to be the center of mass of all points in Ci : ci= 1/Ci * ∑_x∈Ci x
		// basically replace the keys with average of the assigned images
		
		// create the new map that will be returned
		Map<Image, List<Image>> avg_clusters = new HashMap<Image, List<Image>>();
		
		for (Map.Entry<Image, List<Image>> cluster : current_clusters.entrySet()) {
			// calculate the avg of the assigned images for each key
			List<Image> assigned_images = cluster.getValue();
			
			// NOT SURE OF THIS HERE
			if (assigned_images.size() > 0) {
				Image avg_centroid = get_avg_img(assigned_images);
				avg_centroid.set_label(get_dominant_label(assigned_images));
				avg_clusters.put(avg_centroid, assigned_images);
			} else {
				avg_clusters.put(cluster.getKey(), assigned_images);
			}
				
		}
		
		return avg_clusters;
	}
	
	private Image get_avg_img(List<Image> images) {
		List<Double> avg_coords = new ArrayList<>();
		if (images.size() == 0) 
			System.out.println("7chitih");
		
		int dim = images.get(0).get_representation().get_data().size(); // dimension = (1,dim)
		int h_many_imgs = images.size();
		
		for (int i = 0; i < dim; i++) {
			double sum = 0.0;
			for (Image img : images) 
				sum += img.get_representation().get_data().get(i);
			
			avg_coords.add(sum / h_many_imgs);
		}
		
		String representation_type = images.get(0).get_representation().get_name();
			
		return new Image(new Representation(avg_coords, representation_type)) ;
	}
	
	private double squared_dist(Image x, Image y) {
		// squarred euclidean distance
		List<Double> x_data = x.get_representation().get_data();
		List<Double> y_data = y.get_representation().get_data();
		
		assert x_data.size() == y_data.size();
		
		double sum = 0;
		
		if (p == 2)
			for (int i = 0; i < x_data.size(); i++)
				sum += Math.pow((x_data.get(i) - y_data.get(i)), 2);
		else
			for (int i = 0; i < x_data.size(); i++)
				sum += Math.abs((x_data.get(i) - y_data.get(i)));
		
		return sum;
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
	
	private int get_dominant_label(List<Image> images) {
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

}

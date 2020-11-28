package classifiers;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;

import dataset.Dataset;
import dataset.Image;

public class Kmeans extends AbstractClassifier {
	private int k, p; // k for how many clusters, and p for which distance to use
	private long max_iter; // max iterations to halt
	private String init_method; // random or use Kmeans++
	private Dataset training_set;
	private static Random random = new Random(); // class variable serves to generate random data
	
	public Map<Centroid, List<Image>> clusters; // constructed clusters that will be used for predictions
	
	@SuppressWarnings("unused")
	private Kmeans() {} // forcing the user to specify k and p.
	
	public Kmeans(int k, int p) {
		this(k, p, 100, "Kmeans++");
	}
	
	public Kmeans(int k, int p, long max_iter) {
		this(k, p, max_iter, "Kmeans++"); // Using Kmeans++ by default to construct the first centroids
	}
	
	public Kmeans(int k, int p, long max_iter, String init_method) {
		assert k > 0 && p > 0 && max_iter > 0;
		
		this.k = k;
		this.p = p;
		this.max_iter = max_iter;
		this.init_method = init_method;
	}

	@Override
	public boolean train(Dataset training_dataset) {
		this.training_set = training_dataset;
//		assert k <= training_set.size(); 
		// initialization phase
		
		Map<Centroid, List<Image>> initial_clusters = init(this.init_method);
		
		for (int i = 0; i < max_iter; i++) {
			// iter over the dataset to assign images to their nearest centroid
			
			for (Image img : training_dataset.keySet()) {
				Centroid nearest_centroid = get_the_nearest_centroid(img, List.copyOf(initial_clusters.keySet())).getKey(); 
				
				initial_clusters.get(nearest_centroid).add(img);
			}
			
			for (Centroid centroid : initial_clusters.keySet())
				centroid.set_label(get_dominant_label(initial_clusters.get(centroid)));
			
			// calculate the avg clusters
			Map<Centroid, List<Image>> avg_clusters = new HashMap<>();
			
			for (Map.Entry<Centroid, List<Image>> entry : initial_clusters.entrySet()) {
				Centroid avg_centroid = get_new_avg_centroid(entry.getKey(), 
						  									 entry.getValue());
				avg_centroid.set_label(get_dominant_label(entry.getValue()));
				avg_clusters.put(avg_centroid, new ArrayList<>());
			}
			
			// 
			
			initial_clusters = avg_clusters;
		}
		
		this.clusters = initial_clusters;
		
		return true;
	}

	@Override
	public int predict(Image image) {
		return get_the_nearest_centroid(image, List.copyOf(clusters.keySet())).getKey().get_label();
	}

	@Override
	public String toString() {
		
		return "";
	}
	
	private Map<Centroid, List<Image>> init(String method) {
		Map<Centroid, List<Image>> initial_clusters = new HashMap<>();
		List<Centroid> centroids = new ArrayList<>();
		
		if (method.equals("random")) { 		
			// randomly generate the first centroids
			for (int i = 0; i < k; i++) 
				centroids.add(new Centroid(get_random_coords(training_set.get_mins(), 
						  									 training_set.get_maxs())));
		} else {
			
//			// otherwise use kmeans++
//			
//			// 2.2 in http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
//			// initialization phase using Kmeans++ approach: D^2 weighting
//			// constructing the array of centroids
//			
//			// getting the list of all datapoints
//			List<Image> all_datapoints = new ArrayList<>(training_set.keySet());
//			
//			// Take one centeroid c0, chosen uniformly at random from X (all_datapoints)
////			Image selected_img = all_datapoints.get(random.nextInt(all_datapoints.size()));
//			
//			Image selected_img = all_datapoints.get(0); // for debug
//			centroids.add(new Centroid(new ArrayList<>(selected_img.get_representation().get_data())));
//			
//			// remove the selected_img from all_datapoints list so we don't calculate its
//			// distance to all the centroids ( because it's already a centroid)
//			// https://www.youtube.com/watch?v=HatwtJSsj5Q
//			all_datapoints.remove(selected_img);
//			
//			// 1c. Repeat Step 1b. until we have taken k centers altogether
//			for (int i = 1; i < this.k; i++) {
//				// 1b. Take a new center ci, choosing x∈X with probability D(x)^2 / (sum[for x in X] of D(x)^2)
//				
//				// calculating all distances from all datapoints to all current centroidS
//				
//				double sum = 0; // (sum[for x in X] of D(x)^2)
//				List<Double> distances = new ArrayList<>(); // each entry will hold the MIN distance between all centroids and datapoint_i;
//				
//				// filling distances list i.e calculating D(x)^2 for x∈X  
//				for (Image img : all_datapoints) {
//					double dist = get_the_nearest_centroid(img, centroids).getValue();
//					distances.add(dist);
//					sum += dist;
//				}
//				
//				// calculating the proportions/probabilities list, all values now are between 0 and 1
//				List<Double> probabilities = new ArrayList<>();
//				
//				for (Double distance : distances)
//					probabilities.add(distance / sum); // probability = D(x)^2 / (sum[for x in X] of D(x)^2)
//				
//				// Weighted random choice using the probabilities list ( get the index )
//				int chosen_centroid_index = weighted_random(probabilities);
//				System.out.println(chosen_centroid_index);
//				
//				// choosing x∈X with probability D(x)^2 / (sum[for x in X] of D(x)^2)
//				// add the chosen img[i] to centroids list and remove it from datapoints list
//				selected_img = all_datapoints.get(chosen_centroid_index); 
//				centroids.add(new Centroid(new ArrayList<>(selected_img.get_representation().get_data())));
//				all_datapoints.remove(selected_img);
			
			 centroids = kmeans_plus_plus();
//			}
		}
		
		
		for (Centroid centroid : centroids)
			initial_clusters.put(centroid, new ArrayList<>());
		
		return initial_clusters;
	}
	
	private List<Centroid> kmeans_plus_plus() {
		List<Image> all_datapoints = new ArrayList<>(training_set.keySet());
		List<Centroid> centroids = new ArrayList<>();

		// choose a centroid randomly
		int random_index = random.nextInt(all_datapoints.size());
		centroids.add(new Centroid(all_datapoints.get(random_index).get_representation().get_data()));
		
		// remove this datapoint so we don't choose it again
		all_datapoints.remove(all_datapoints.get(random_index));
		
		// the rest choose them smartly
		for (int i = 1; i < this.k; i++) {
			double max_dist = Double.NEGATIVE_INFINITY;
			int max_index = 0;
			
			for (int j = 0; j < all_datapoints.size(); j++) {
//				System.out.println("For image " + j + " with coords" + all_datapoints.get(j).get_representation().get_data());
//				System.out.println("Its nearest centroid is " + get_the_nearest_centroid(all_datapoints.get(j), centroids).getKey().get_coords());
				double dist = get_the_nearest_centroid(all_datapoints.get(j), centroids).getValue();
//				System.out.println("With a distance of " + dist + "\n");
				
				if (dist > max_dist) {
					max_dist = dist;
					max_index = j;
				}
			}
			
//			System.out.println("next centroid will be " + all_datapoints.get(max_index).get_representation().get_data());
			// make the far as the new centroid
			centroids.add(new Centroid(all_datapoints.get(max_index).get_representation().get_data()));
			all_datapoints.remove(all_datapoints.get(max_index));
		}
		
		return centroids;
	}
	
	private Map.Entry<Centroid, Double> get_the_nearest_centroid(Image img, List<Centroid> centroids) {
		// return the nearest centroid relative to the passed img
		assert centroids.size() > 0;
		
		List<Double> img_coords = img.get_representation().get_data();
		
		
		int min_index = 0, size = centroids.size();
		double min_dist = dist(img_coords, centroids.get(0).get_coords(), this.p);
		
		for (int i = 1; i < size; i++) {
			double new_dist = dist(img_coords, centroids.get(i).get_coords(), this.p);
			
			if (new_dist < min_dist) {
				min_dist = new_dist;
				min_index = i;
			}
		}

		return new AbstractMap.SimpleImmutableEntry<>(centroids.get(min_index), min_dist);
	}
	
	private Centroid get_new_avg_centroid(Centroid centroid, List<Image> assigned_images) {
		// this method will calculate a new centroid, by averaging all the assigned images
		// and assign the same label to it
		
		if (assigned_images == null || assigned_images.size() == 0)
			return centroid;
		
//		if (this.assigned_images.size() == 0)
//			System.err.println("Assigned imgs list in centroid is empty !");
		
		List<Double> avg_coords = new ArrayList<>();
		
		int dim = assigned_images.get(0).get_representation().get_data().size();
		int h_many_imgs = assigned_images.size();
		
		for (int i = 0; i < dim; i++) {
			double sum = 0;
			
			for (Image img : assigned_images) 
				sum += img.get_representation().get_data().get(i);
			
			avg_coords.add(sum / h_many_imgs);
		}
		
		// CHECK THIS
		Centroid avg_centroid = new Centroid(avg_coords, centroid.get_label());
		
		return avg_centroid;
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
	
	private List<Double> get_random_coords(List<Double> mins, List<Double> maxs) {
		// return a list of random coordinates
		// each coords_i will be ranged from mins.get(i) to maxs.get(i)
		
		assert mins.size() > 0 && mins.size() == maxs.size();
		
		List<Double> random_coords = new ArrayList<>();
		
		for (int i = 0; i < mins.size(); i++) {
			double min = mins.get(i);
			double max = maxs.get(i);
			
			random_coords.add(random.nextDouble() * (max - min) + min);
		}
		
		return random_coords;
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
}

package dataset;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class Dataset {
	/*
	 * Dataset class will contain a map of (Image, label (int))
	 * 
	 * Dataset needs to be coherent ( Only one type of representation can be used )
	 * 
	 * Type of representation that will be used, will be determined by the first
	 * inserted image.
	 */
	
	private Map<Image, Integer> dataset;
	private String representation_type;
	private boolean first_datapoint; // useful for the next insertion after the rep_type is determined
	
	/*
	 * very useful for Kmeans random initialization step, keeping the range = (min, max) 
	 * for each feature // so we can generate the first centroids accordingly
	 * 
	 * So now each feature i will range between [mins.get(i); maxs.get(i)] 
	 */	
	
	private List<Double> mins, maxs; // will be initialized in the first insertion
	
	public Dataset() {
		dataset = new HashMap<>();
		representation_type = "";
		first_datapoint = true;
	}
	
	public Dataset(String dir_path) throws IOException {
		this();
		load_dataset_from_directory(dir_path);
	}
	
	public Map<Image, Integer> get_dataset() {
		return dataset;
	}
	
	public String get_representation_type() {
		return representation_type;
	}
	
	public List<Double> get_mins() {
		return mins;
	}
	
	public List<Double> get_maxs() {
		return maxs;
	}
	
	public boolean add_datapoint(Image img, int label) {
		assert label >= 1 && label <= 9;
		
		if (first_datapoint) {
			dataset.put(img, label);
			representation_type = img.get_representation_type();
			
			first_datapoint = false;
			
			mins = new ArrayList<>(img.get_representation().get_data());
			maxs = new ArrayList<>(img.get_representation().get_data());
			
			return true;
		}
		else {
			if (img.get_representation_type().equals(representation_type)) {
				if (!dataset.containsKey(img)) {
					dataset.put(img, label);
					update_range(img);
					return true;
				}
				else 
					System.err.println("Image already in dataset !");
			}
			else 
				System.err.println("Passed image is not represented the same way !"
							+ " [dataset accepts only " + representation_type + " representation type ]");
		}
		
		return false;
	}
	
	public boolean remove_datapoint(Image img) {
		if (dataset.containsKey(img)) {
			dataset.remove(img);
			return true;
		}
		
		return false;
	}
	
	public void reset() {
		dataset.clear();
		representation_type = "";
		first_datapoint = true;
	}
	
	public void load_dataset_from_directory(String dir_path) throws IOException {
		// reset the dataset and load all the images ( their representations to be more accurate )
		// to RAM.
		
		reset();
		
		List<File> filesInFolder = Files.walk(Paths.get(dir_path), 1) // 1 is depth 
								        .filter(Files::isRegularFile)
								        .map(Path::toFile)
								        .collect(Collectors.toList());
		Image new_img;
		
		if (filesInFolder.size() == 0)
			throw new IOException("Empty directory");
		
		for (File file : filesInFolder) 	
			try {
				new_img = new Image(new Representation(file.getAbsolutePath()));
				add_datapoint(new_img, new_img.get_label());
				
			} catch (BadRepresentationFileException e) {
				System.err.println("Skipping " + file.getName() + " -> Bad format");
			}	
		
	}
	
	// Delegate
	
	public Set<Image> keySet() {
		return dataset.keySet();
	}
	
	public Set<Map.Entry<Image,Integer>> entrySet() {
		return dataset.entrySet();
	}
	
	public int get(Image img) {
		return dataset.get(img);
	}
	
	public int size() {
		return dataset.size();
	}
	
	public boolean isEmpty() {
		return dataset.isEmpty();
	}
	
	private void update_range(Image img) {
		/*
		 * Update the range for each feature ( min, max )
		 */
		List<Double> img_data = img.get_representation().get_data();
		
		for (int i = 0; i < img_data.size(); i++) {
			if(img_data.get(i) < mins.get(i)) 
				mins.set(i, img_data.get(i));
			
			if(img_data.get(i) > maxs.get(i)) 
				maxs.set(i, img_data.get(i));
		}
	}
}

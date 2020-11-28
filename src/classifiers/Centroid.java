package classifiers;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;

import dataset.Image;

public class Centroid {
	private int label; // the dominant label of the cluster where this centroid exists
	private List<Double> coords; // coordinates of this centroid
	
	@SuppressWarnings("unused")
	private Centroid() {} // Centroid must have coords, so no default construct for the user

	public Centroid(List<Double> coords) {
		this.label = -100;
		this.coords = coords;
	}
	
	public Centroid(List<Double> coords, int label) {
		this.coords = coords;
		this.label = label;
	}
	
	public void set_label(int label) {
		this.label = label;
	}
	
	public int get_label() {
		return label;
	}
	
	public List<Double> get_coords() {
		return coords;
	}
	
	
//	public Centroid get_new_avg_centroid() {
//		// this method will calculate a new centroid, by averaging all the assigned images
//		// and assign the same label to it
//		
//		if (assigned_images == null || assigned_images.size() == 0)
//			return this;
//		
////		if (this.assigned_images.size() == 0)
////			System.err.println("Assigned imgs list in centroid is empty !");
//		
//		List<Double> avg_coords = new ArrayList<>();
//		
//		int dim = this.assigned_images.get(0).get_representation().get_data().size();
//		
//		for (int i = 0; i < dim; i++) {
//			double sum = 0;
//			
//			for (Image img : this.assigned_images)
//				sum += img.get_representation().get_data().get(i);
//			
//			avg_coords.add(sum / dim);
//		}
//		
//		Centroid avg_centroid = new Centroid(avg_coords, this.assigned_images);
//		avg_centroid.set_label(label);
//		
//		return avg_centroid;
//	}
	
	public Centroid clone() {
		Centroid c = new Centroid(new ArrayList<Double>(coords), label);
		return c;
	}
	
	@Override
	public String toString() {
		return " -> label : " + label + " Centroid_coords: " + coords + "\n";
//		return " -> label : " + label;
	}
	
	@Override
	public int hashCode() {
		return coords.hashCode();
	}
	
	@Override
	public boolean equals(Object o) {
		if (!(o == this) || ! (o instanceof Centroid))
			return false;
		
		Centroid other = (Centroid) o;
		
		return other.coords.equals(this.coords) && other.label == this.label;
	} 

}

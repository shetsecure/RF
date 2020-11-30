package classifiers;

import java.util.ArrayList;
import java.util.List;

public class Centroid {
	/*
	 * Centroid class representing the centroids used by Kmeans classifier.
	 */
	
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
	
	public Centroid clone() {
		Centroid c = new Centroid(new ArrayList<Double>(coords), label);
		return c;
	}
	
	@Override
	public String toString() {
		return " -> label : " + label + " Centroid_coords: " + coords + "\n";
	}
	
	@Override
	public int hashCode() {
		return coords.hashCode();
	}
	
	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		
		if (! (o instanceof Centroid))
			return false;
		
		Centroid other = (Centroid) o;
		
		return other.coords.equals(this.coords) && other.label == this.label;
	} 

}

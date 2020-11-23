import java.util.HashMap;
import java.util.Map;

public class Image {
	/*
	 * An image have at most 1 representation. 
	 * It indeed can be represented by multiple representations, but we forced one representation to create
	 * a homogeneous dataset ( we need to train on the same TYPE of representations )
	 * 
	 * An image object will have essentially a representation and a label assigned to it.
	 * 
	 */
	
	private Representation rep;
	private String img_path, representation_type; // img_path will be useful in the future for GUI
	private int label; // which class 
	
	// a class variable that is shared in all instances, to know the corresponding class given a label.
	@SuppressWarnings("serial")
	public static final Map<Integer, String> label_map = new HashMap<Integer, String>() {{
	    put(1, "Poisson");
	    put(2, "Lapin");
	    put(3, "silhouette");
	    put(4, "avion");
	    put(5, "main");
	    put(6, "outil");
	    put(7, "UNKNOWN_7"); // need to be specified in the PDF
	    put(8, "animal");
	    put(9, "UNKNOWN_9"); // need to be specified in the PDF
	}}; 
	
	public Image(Representation rep) {
		this.label = -1; // no assigned label yet
		this.rep = rep;
		this.representation_type = rep.get_name();
		this.img_path = "";
		
		conclude_label_from_filename(); // try to conclude the label if we can
	}
	
	public Image(Representation rep, int label) {
		this(rep);
		
		assert label >= 1 && label <= 9;
		this.label = label;
	}
	
	public Image(Representation rep, int label, String img_path) {
		this(rep, label);
		this.img_path = img_path;
	}
	
	public String get_path() {
		return img_path;
	}
	
	public int get_label() {
		return label;
	}
	
	public String get_class() {
		return label_map.get(this.label);
	}
	
	public String get_representation_type() {
		return representation_type;
	}
	
	public Representation get_representation() {
		return rep;
	}
	
	private boolean conclude_label_from_filename() {
		// conclude label from filename
		// assuming filename format: SxxNyyy.*
		// xx is the class ( range = [[ 0, 9 ]]
		
		String filename = rep.get_filename(); // filename format: SxxNyyy.*
		
		if (filename.length() > 3) {
			int label = Integer.parseInt(filename.substring(1, 3)); // xx
			
			if (label < 1 || label > 9) 
				System.err.println("WARNING, filename implied a new label of " + label);
			else {
				this.label = label;
				return true;
			}
		}
		else 
			System.err.println("Filename is short [" + filename + "]");
		
		return false;
	}
	
	@Override
	public String toString() {
		return "[+] class: " +  get_class() + " | label : " + label + "  | Filename: " + rep.get_filename();
	}
	
	@Override
	public boolean equals(Object o) {
		
		if (o == this) 
			return true;
		
		if (! (o instanceof Image) || o == null) 
			return false;
		
		Image other = (Image) o;
		
		// two images are the same if they are represented in the same way
		// and they have the same attached label to them.
		
		return other.get_representation().equals(this.rep) && 
			   other.get_label() == this.label;
		
	}
	
	// Overrided hashCode because it will be useful in Dataset class
	// to determine if a given image is already in the dataset or not
	// so that we don't include duplicates ( useless )
	@Override
	public int hashCode() {
		return rep.get_data().hashCode();
	}
}

import java.util.Map;
import java.util.HashMap;

public class Image {
	private Representation rep;
	private String img_path, representation_name;
	private int label; // which class 
	
	@SuppressWarnings("serial")
	public static final Map<Integer, String> label_map = new HashMap<Integer, String>() {{
	    put(1, "Poisson");
	    put(2, "Lapin");
	    put(3, "silhouette");
	    put(4, "avion");
	    put(5, "main");
	    put(6, "outil");
	    put(7, "UNKNOWN_7");
	    put(8, "animal");
	    put(9, "UNKNOWN_9");
	}};
	
	public Image(Representation rep) {
		this.label = -1;
		this.rep = rep;
		this.representation_name = rep.get_name();
		this.img_path = "";
		
		conclude_label_from_filename();
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
	
	public String get_representation_name() {
		return representation_name;
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
				System.out.println("WARNING, filename implied a new label of " + label);
			else {
				this.label = label;
				return true;
			}
		}
		else {
			System.out.println("Filename is shot ! check filename");
		}
		return false;
	}
}

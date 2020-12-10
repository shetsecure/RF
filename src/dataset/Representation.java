package dataset;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Representation {
	/*
	 * This class portrays the image's representation (E34, GFD, SA or F0).
	 * It will provide useful methods to access the data ( vector of numeric values )
	 * 
	 * Tasks:
	 * 		- Read a representation file
	 * 		- Validate a representation file
	 * 		- Getting the filename ( will be useful to conclude the label if the SxxNyyy format is respected )
	 * 		
	 */
	
	private String name, file_path, filename;
	private List<Double> data;
	
	private Representation() {}
	
	public Representation(String file_path) throws BadRepresentationFileException{
		// we can conclude which representation is it by just knowing
		// how many values are there in the corresponding file.
		// Independent from the file's extension.
		
		data = read_representation(file_path);
		this.filename = "";
		
		if (data != null) {
			this.file_path = file_path;
			construct_filename_from_path();
			
			switch (data.size()) {
				case 16:
					name = "E34";
					break;
				case 90:
					name = "SA";
					break;
				case 100:
					name = "GFD";
					break;
				case 128:
					name = "F0";
					break;
				default:
					throw new BadRepresentationFileException("Unknown representation !");
			}
		}
		else
			throw new BadRepresentationFileException("Couldnt read the file !");
	}
	
	public Representation(List<Double> data, String name) {
		this.data = data;
		this.name = name;
	}
	
	public Representation(Double[] data, String name) {
		this.data = Arrays.asList(data);
		this.name = name;
	}
	
	public String get_name() {
		return name;
	}
	
	public String get_file_path() {
		return file_path;
	}
	
	public String get_filename() {
		return filename;
	}
	
	public List<Double> get_data() {
		return data;
	}

	public List<Double> read_representation(String path){
		// read a representation file ( List of numbers )
		// returns a list of double or null if we couldn't.
		
		try {
			List<Double> data = Files.lines(Paths.get(path))
	                				 .map(Double::parseDouble)
	                				 .collect(Collectors.toList());
			return data;
		} catch (NoSuchFileException e) {
			System.err.println("File not found !");
		    e.printStackTrace();
		} catch (IOException e) {
			System.err.println(e.getMessage());
		    e.printStackTrace();
		}
		
		return null;
	}
	
	private void construct_filename_from_path() {
		Pattern p = Pattern.compile("s[0-9][0-9]n[0-9][0-9][0-9][.]", Pattern.CASE_INSENSITIVE);
		// added Pattern.CASE_INSENSITIVE, because an example of failure: S02n005.GFD [ uppercase S ]
        Matcher m = p.matcher(this.file_path);
        
        if (m.find()) {
        	String match = m.group();
        	
        	// to remove the dot at the end
        	this.filename = match.substring(0, match.length() - 1); 
        }
	}
	
	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		
		if (! (o instanceof Representation) || o == null)
			return false;
		
		Representation other = (Representation) o;
		
		// Two representations are the same if they have the same list of numbers
		// and the same type of representation.
		
		return other.get_data().equals(this.data) &&
			   other.get_name().equals(this.name);
	}
	
	public Representation clone() {
		Representation rep = new Representation();
		
		rep.name = this.name;
		rep.file_path = this.file_path;
		rep.filename = this.filename;
		rep.data = new ArrayList<Double>(this.data);
		
		return rep;
	}
} 

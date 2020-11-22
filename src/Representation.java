import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Representation {	
	private String name, file_path, filename;
	private List<Double> data;
	
	public Representation(String file_path) throws BadRepresentationFileException{
		// we can conclude which representation is it by just knowing
		// how many values are there in the corresponding file
		
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
				case 100:
					name = "GFD";
				case 128:
					name = "F0";
				default:
					throw new BadRepresentationFileException("Unknown representation !");
			}
		}
		else
			throw new BadRepresentationFileException("Couldnt read the file !");
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
		// returns a list of double 
		
		try {
			List<Double> data = Files.lines(Paths.get(path))
	                				 .map(Double::parseDouble)
	                				 .collect(Collectors.toList());
			return data;
		} catch (NoSuchFileException e) {
			System.out.println("File not found !");
		    e.printStackTrace();
		} catch (IOException e) {
			System.out.println(e.getMessage());
		    e.printStackTrace();
		}
		
		return null;
	}
	
	private void construct_filename_from_path() {
		Pattern p = Pattern.compile("s[0-9][0-9]n[0-9][0-9][0-9][.]");
        Matcher m = p.matcher(this.file_path);
        
        if (m.find()) {
        	String match = m.group();
        	
        	// to remove the dot at the end
        	this.filename = match.substring(0, match.length() - 1); 
        }  	 
	}
}

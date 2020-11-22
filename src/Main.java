import java.util.Map;

public class Main {

	public static void main(String[] args) {
//		try {
//			Image img = new Image(new Representation("project_files/E34/s01n008.E34"));
//			Image img2 = new Image(new Representation("project_files/E34/s01n008.E34"));
//			System.out.println(img.equals(img2));
//			System.out.println(img.get_label());
//			System.out.println(img.get_class());
//			System.out.println(img.get_representation_name());
//			System.out.println(img.get_representation().get_data());
//			System.out.println(img.get_representation().get_filename());
//		} catch (BadRepresentationFileException e) {
//			e.printStackTrace();
//		}
		
		Dataset d = new Dataset("project_files/SA");
		try {
			d.add_datapoint(new Image(new Representation("project_files/SA/s01n005.SA")), 2);
			d.add_datapoint(new Image(new Representation("project_files/F0/s01n004.F0")), 2);
		} catch (BadRepresentationFileException e) {
			e.printStackTrace();
		}
		
		for(Map.Entry<Image,Integer> entry : d.entrySet()) 
			System.out.println(entry.getKey() + " -> Label: " + entry.getValue());
		
		System.out.println("Size of dataset: " + d.size());
	}

}

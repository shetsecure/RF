
public class Main {

	public static void main(String[] args) {
		try {
			Image img = new Image(new Representation("project_files/E34/s01n008.E34"));
			System.out.println(img.get_label());
			System.out.println(img.get_class());
			System.out.println(img.get_representation_name());
			System.out.println(img.get_representation().get_data());
			System.out.println(img.get_representation().get_filename());
		} catch (BadRepresentationFileException e) {
			e.printStackTrace();
		}
		
	}

}

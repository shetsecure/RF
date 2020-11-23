import java.util.Map.Entry;

public abstract class AbstractClassifier {
	/* Abstract class that will serve as a blueprint to all classifiers
	 * Each classifier needs to be trained on the dataset
	 * Each classifier must have a predict method
	 * Most classifiers have a some kind of scoring method to calculate in order to classify
	
	 * accuracy is the same for all of them, so it is implemented here.
	 *  
	 */
	
	protected abstract boolean train(Dataset training_dataset);
	
	protected abstract double score(Image img);
	
	protected abstract int predict(Image img);
	
	protected double accuracy(Dataset test_dataset) {
		int counter = 0, size = test_dataset.size();
		
		for(Entry<Image, Integer> entry : test_dataset.entrySet()) {
			int predicted_label = predict(entry.getKey());
			
			if ( predicted_label == entry.getValue())
				counter++;
//			else {
//				System.err.println(entry.getKey());
//				System.err.println("Classifier predicted: " + predicted_label 
//						+ "The real label was " + entry.getKey().get_label());
//			}
		}
		
		// casting counter to double to have a floating point division
		
		return (double) counter / size;
	}
	
	@Override
	public abstract String toString();
}

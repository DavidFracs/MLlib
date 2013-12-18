package classification;

import data.Dataset;
import data.Instance;

public interface Classifier 
{
	public void buildModel(Dataset dataset);
	public void predict(Dataset dataset);
	public void predict(Instance inst);
}

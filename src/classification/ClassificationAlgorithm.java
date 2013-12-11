package classification;

import data.Dataset;
import data.Instance;

public interface ClassificationAlgorithm 
{
	public void buildModel(Dataset dataset);
	public void predict(Dataset dataset);
	public void predict(Instance inst);
}

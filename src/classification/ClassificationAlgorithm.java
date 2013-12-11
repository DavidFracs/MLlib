package classification;

import data.Dataset;

public interface ClassificationAlgorithm 
{
	public void buildModel(Dataset dataset);
}

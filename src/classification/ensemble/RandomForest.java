package classification.ensemble;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import classification.Classifier;
import classification.tree.DecisionTreeC45;
import data.Dataset;
import data.Instance;
import data.Instance.InstanceType;
import data.SparseDataset;
import data.SparseInstance;

public class RandomForest implements Classifier 
{
	private DecisionTreeC45[] estimators = null;
	private Random random = new Random();
	
	public int estimatorCount = 301;
	public double sampleFeaturePercent = 0.15;
	public int minFeatureCount = 5;
	public int minLeafSize = 2;
	
	@Override
	public void buildModel(Dataset dataset) 
	{
		estimators = new DecisionTreeC45[estimatorCount];
		for(int i = 0; i < estimatorCount; i++)
		{
			estimators[i] = new DecisionTreeC45();
			estimators[i].minLeafSize = minLeafSize;
			SparseDataset sampledData = sampleData(dataset);
			estimators[i].buildModel(sampledData);
		}
	}

	private SparseDataset sampleData(Dataset dataset) 
	{
		SparseDataset sampledData = new SparseDataset();
		
		int featureCount = (int) (dataset.featureCount * sampleFeaturePercent);
		featureCount = minFeatureCount > featureCount ? minFeatureCount : featureCount;
		sampledData.featureCount = dataset.featureCount;
		sampledData.feature2Type = dataset.feature2Type;
		HashSet<Integer> sampledFeatures = new HashSet<Integer>();
		while(sampledFeatures.size() < featureCount)
			sampledFeatures.add((int)(random.nextDouble() * dataset.featureCount));
			
		int instCount = dataset.trainCount;
		HashMap<Integer, Integer> chooser = new HashMap<Integer, Integer>();
		for(int i = 0; i < instCount; i++)
		{
			int instId = (int)(random.nextDouble() * instCount);
			if(!chooser.containsKey(instId))
				chooser.put(instId, 1);
			else 
				chooser.put(instId, 1 + chooser.get(instId));
		}
		int nn = -1;
		for(Instance inst : dataset.data)
		{
			if(inst.type != InstanceType.Train) continue;
			if(!chooser.containsKey(++nn)) continue;
			SparseInstance newInst = new SparseInstance(inst.id, inst.target);
			newInst.type = inst.type;
			for(int fid : sampledFeatures)
			{
				if(inst.containsFeature(fid))
					newInst.addFeature(fid, inst.getFeature(fid));
			}
			for(int i = 0; i < chooser.get(nn); i++)
				sampledData.addInstance(newInst);
		}
		return sampledData;
	}

	@Override
	public void predict(Dataset dataset) 
	{
		for(Instance inst : dataset.data)
		{
			predict(inst);
		}
	
	}

	@Override
	public void predict(Instance inst) 
	{
		double positive = 0;
		for(int i = 0; i < estimatorCount; i++)
		{
			estimators[i].predict(inst);
			positive += inst.predict;
		}
		if(positive > estimatorCount - positive)
			inst.predict = 1.0;
		else
			inst.predict = 0.0;
	}

}

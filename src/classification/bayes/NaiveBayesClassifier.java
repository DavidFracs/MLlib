package classification.bayes;

import java.util.HashMap;

import classification.Classifier;
import data.Dataset;
import data.Instance;
import data.Instance.InstanceType;

public class NaiveBayesClassifier implements Classifier 
{
	public double smoothParam = Math.log(0.00001);
	
	private HashMap<Integer, Double> classProb = new HashMap<Integer, Double>();
	private HashMap<Integer, HashMap<Integer, Double>> classFeatureProb = new HashMap<Integer, HashMap<Integer, Double>>();
	
	@Override
	public void buildModel(Dataset dataset) 
	{
		smoothParam = Math.log(1.0 / dataset.trainCount);
		HashMap<Integer, Integer> classInstCount = new HashMap<Integer, Integer>();
		HashMap<Integer, Double> classTotalFeatureCount = new HashMap<Integer, Double>();
		HashMap<Integer, HashMap<Integer, Double>> classFeatureCount = new HashMap<Integer, HashMap<Integer, Double>>();
		for(Instance inst : dataset.data)
		{
			if(inst.type != InstanceType.Train) continue;
			int c = (int)inst.target;
			if(!classInstCount.containsKey(c))
			{
				classInstCount.put(c, 0);
				classTotalFeatureCount.put(c, 0.0);
				classFeatureCount.put(c, new HashMap<Integer, Double>());
			}
			classInstCount.put(c, 1 + classInstCount.get(c));
			int[] features = inst.getFeatureIds();
			for(int i = 0; i < features.length; i++)
			{
				if(!classFeatureCount.get(c).containsKey(features[i]))
					classFeatureCount.get(c).put(features[i], 0.0);
				classFeatureCount.get(c).put(features[i], inst.getFeature(features[i]) + classFeatureCount.get(c).get(features[i]));
				classTotalFeatureCount.put(c, inst.getFeature(features[i]) + classTotalFeatureCount.get(c));
			}
		}
		
		for(int c : classInstCount.keySet())
		{
			classProb.put(c, Math.log(classInstCount.get(c) * 1.0 / dataset.trainCount));
			classFeatureProb.put(c, new HashMap<Integer, Double>());
			for(int fid : classFeatureCount.get(c).keySet())
				classFeatureProb.get(c).put(fid, Math.log((1.0 + classFeatureCount.get(c).get(fid)) / (classTotalFeatureCount.get(c) + dataset.featureCount)));
		}
		classInstCount.clear();
		classFeatureCount.clear();
		classTotalFeatureCount.clear();
	}

	@Override
	public void predict(Dataset dataset) 
	{
		for(Instance inst : dataset.data)
			predict(inst);
	}

	@Override
	public void predict(Instance inst) 
	{
		HashMap<Integer, Double> probs = new HashMap<Integer, Double>();
		int[] features = inst.getFeatureIds();
		for(int i = 0; i < features.length; i++)
		{
			double value = inst.getFeature(features[i]);
			for(int c : classProb.keySet())
			{
				if(classFeatureProb.get(c).containsKey(features[i]))
				{
					if(!probs.containsKey(c))
						probs.put(c, 0.0);
					probs.put(c, probs.get(c) + value * (classFeatureProb.get(c).get(features[i]) - this.smoothParam));
				}
			}
		}
		double maxProb = Double.MIN_VALUE;
		int maxClass = -1;
		for(int c : probs.keySet())
		{
			double p = classProb.get(c) + probs.get(c);
			if(p > maxProb)
			{
				maxProb = p;
				maxClass = c;
			}
		}
		inst.predict = maxClass;
	}
}

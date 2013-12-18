package classification.bayes;

import java.util.HashMap;

import classification.Classifier;
import data.Dataset;
import data.Instance;
import data.Instance.InstanceType;

public class RevisedNaiveBayesClassifier implements Classifier 
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
		HashMap<Integer, Double> featureCount = new HashMap<Integer, Double>();
		HashMap<Integer, HashMap<Integer, Double>> classFeatureCount = new HashMap<Integer, HashMap<Integer, Double>>();
		double totalFeatureCount = 0;
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
				double value = inst.getFeature(features[i]);
				if(!featureCount.containsKey(features[i]))
					featureCount.put(features[i], 0.0);
				featureCount.put(features[i], value + featureCount.get(features[i]));
				totalFeatureCount += value;
				if(!classFeatureCount.get(c).containsKey(features[i]))
					classFeatureCount.get(c).put(features[i], 0.0);
				classFeatureCount.get(c).put(features[i], value + classFeatureCount.get(c).get(features[i]));
				classTotalFeatureCount.put(c, inst.getFeature(features[i]) + classTotalFeatureCount.get(c));
			}
		}
		
		for(int c : classInstCount.keySet())
		{
			classProb.put(c, Math.log(classInstCount.get(c) * 1.0 / dataset.trainCount));
			classFeatureProb.put(c, new HashMap<Integer, Double>());
			
			double countExcludeC = totalFeatureCount - classTotalFeatureCount.get(c) + dataset.featureCount;
			double total = 0;
			for(int fid : featureCount.keySet())
			{
				double featureCountExcludeC = featureCount.get(fid) + 1;
				if(classFeatureCount.get(c).containsKey(fid))
					featureCountExcludeC -= classFeatureCount.get(c).get(fid);
				classFeatureProb.get(c).put(fid, Math.log(featureCountExcludeC / countExcludeC));
				total += Math.log(featureCountExcludeC / countExcludeC);
			}
			//total = 0 - total;
			//for(int fid : classFeatureProb.get(c).keySet())
			//{
			//	classFeatureProb.get(c).put(fid, classFeatureProb.get(c).get(fid) / total);
			//}
		}
		classInstCount.clear();
		classFeatureCount.clear();
		classTotalFeatureCount.clear();
	}

	public void predict(Dataset dataset) 
	{
		for(Instance inst : dataset.data)
			predict(inst);
	}

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
					probs.put(c, probs.get(c) + value * (0 - classFeatureProb.get(c).get(features[i])));
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

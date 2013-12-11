package data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import data.Feature.FeatureType;
import data.Instance.InstanceType;

public abstract class Dataset 
{
	public ArrayList<Instance> data = new ArrayList<Instance>();
	public HashMap<Integer, FeatureType> feature2Type = new HashMap<Integer, FeatureType>();
	public HashMap<Integer, String> feature2Name = new HashMap<Integer, String>();
	
	public int featureCount = 0;
	public int trainCount = 0;
	public int testCount  = 0;
	public int quizCount = 0;
	
	private Random seed = new Random();
	
	public void addInstance(Instance inst)
	{
		data.add(inst);
		if(inst.type == InstanceType.Train)
			trainCount++;
		else if(inst.type == InstanceType.Test)
			testCount++;
		else
			quizCount++;
	}
	
	public void randomTrainSet(double trainPercent)
	{
		randomTrainSet(trainPercent, 1 - trainPercent);
	}
	
	public FeatureType getFeatureType(int fid)
	{
		if(feature2Type.containsKey(fid))
			return feature2Type.get(fid);
		return FeatureType.Continuous;
	}
	
	public void randomTrainSet(double trainPercent, double testPercent)
	{
		if(trainPercent < 0 || testPercent < 0)
		{
			System.err.print("Wrong train or test percentages. They should both be larger than zero");
			return;
		}
		trainCount = 0;
		testCount = 0;
		quizCount = 0;
		for(Instance inst : data)
		{
			double r = seed.nextDouble();
			if(r <= trainPercent)
			{
				inst.type = InstanceType.Train;
				trainCount++;
			}
			else if(r <= trainPercent + testPercent)
			{
				inst.type = InstanceType.Test;
				testCount++;
			}
			else
			{			
				inst.type = InstanceType.Quiz;
				quizCount++;
			}
		}
		System.out.println(this.toString());
	}
	
	public String toString()
	{
		return String.format("Instance Count %d %d %d %d", trainCount, testCount, quizCount, data.size());
	}
	
	public abstract void loadFromFile(String trainFile, String testFile, String quizFile, String featureFile);
}

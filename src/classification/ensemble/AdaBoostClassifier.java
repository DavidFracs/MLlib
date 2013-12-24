package classification.ensemble;

import classification.Classifier;
import classification.tree.DecisionTreeC45;
import data.Dataset;
import data.Instance;
import data.Instance.InstanceType;

///////////////////////////////////////////////////////

//Question? how to avoid always choose the same feature?

///////////////////////////////////////////////////////
public class AdaBoostClassifier implements Classifier 
{
	public int estimatorCount = 100;
	public int splitCount = 2;
	
	private DecisionTreeC45[] estimators = null;
	private double[] estimatorWeight = null;
	@Override
	public void buildModel(Dataset dataset) 
	{
		estimators = new DecisionTreeC45[estimatorCount];
		estimatorWeight = new double[estimatorCount];
		for(int i = 0; i < estimatorCount; i++)
		{
			estimators[i] = new DecisionTreeC45();
			estimators[i].maxHeight = splitCount;
			estimators[i].needPrune = false;
			estimators[i].buildModel(dataset);
			double error = 0, sum = 0;
			for(int j = 0; j < dataset.data.size(); j++)
			{
				if(dataset.data.get(j).type != InstanceType.Train)
					continue;
				estimators[i].predict(dataset.data.get(j));
				if(dataset.data.get(j).predict != dataset.data.get(j).target)
					error += dataset.data.get(j).weight;
				sum += dataset.data.get(j).weight;
			}
			double errorRate = error / sum;
			estimatorWeight[i] = Math.log((1-errorRate)/errorRate);
			for(int j = 0; j < dataset.data.size(); j++)
			{
				if(dataset.data.get(j).type != InstanceType.Train)
					continue;
				if(dataset.data.get(j).predict != dataset.data.get(j).target)
					dataset.data.get(j).weight *= Math.exp(estimatorWeight[i]/2);
				else
					dataset.data.get(j).weight /= Math.exp(estimatorWeight[i]/2);
			}
			System.out.println(estimatorWeight[i]);
		}
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
	public void predict(Instance inst) {
		double positive = 0;
		double negative = 0;
		for(int i = 0; i < estimatorCount; i++)
		{
			estimators[i].predict(inst);
			if(inst.predict == 1.0)
				positive += estimatorWeight[i];
			else
				negative += estimatorWeight[i];
		}
		if(positive > negative)
			inst.predict = 1.0;
		else
			inst.predict = 0.0;
		//System.out.println(positive + " " + negative + " " + inst.target);
	}

}

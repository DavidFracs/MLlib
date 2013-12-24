package classification.linear;

import java.util.Random;

import classification.Classifier;
import data.Dataset;
import data.Instance;

public class LogisticRegression implements Classifier 
{
	private double[] w = null;
	
	public double tol = 0.001;
	public double maxStep = 50;
	//public int miniBatch = 1;
	public double learningRate = 0.1;
	public double regWeight = 0.05;
	public double decreaseRate = 0.9;
	public String regMethod = "l1";
	public boolean quiteMode = true;
	
	public void buildModel(Dataset dataset) 
	{
		//initialize
		Random seed = new Random();
		w = new double[dataset.featureCount];
		for(int i = 0; i < w.length; i++)
			w[i] = seed.nextDouble() - 0.5;
		//train
		int step = 0;
		double lastError = Double.MAX_VALUE;
		while(step++ < maxStep)
		{
			double error = 0;
			for(Instance inst : dataset.data)
			{
				double p = 0;
				for(int fid :inst.getFeatureIds())
					p = p + w[fid]*inst.getFeature(fid);
				p = logistic(p);
				for(int fid :inst.getFeatureIds())
				{
					if(regMethod.equals("l1"))
					{
						if(w[fid] > 0)
						{
							w[fid] = w[fid] + learningRate * (inst.getFeature(fid) * (inst.target - p) - regWeight);
							if(w[fid] < 0) w[fid] = 0;
						}
						else if(w[fid] < 0)
						{
							w[fid] = w[fid] + learningRate * (inst.getFeature(fid) * (inst.target - p) + regWeight);
							if(w[fid] > 0) w[fid] = 0;
						}
						else
							w[fid] = w[fid] + learningRate * inst.getFeature(fid) * (inst.target - p);
					}
					else if(regMethod.equals("l2"))
						w[fid] = w[fid] + learningRate * (inst.getFeature(fid) * (inst.target - p) - regWeight * w[fid]);
					else
						w[fid] = w[fid] + learningRate * inst.getFeature(fid) * (inst.target - p);
				}
				error += Math.abs(inst.target - p);
			}
			if(!quiteMode)
				System.out.println(String.format("Step %d : %.9f", step, error));
			if(lastError - error < tol) break;
			lastError = error;
			learningRate = learningRate * decreaseRate;
		}
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
		double p = 0;
		for(int fid :inst.getFeatureIds())
			p = p + w[fid]*inst.getFeature(fid);	
		inst.predict = logistic(p);
	}

	private double logistic(double p)
	{
		return 1.0 / (1 + Math.exp(0 - p));
	}
	
}

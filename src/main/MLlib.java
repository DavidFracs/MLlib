package main;

import java.io.FileOutputStream;
import java.io.PrintStream;

import classification.tree.DecisionTreeC45;
import data.Dataset;
import data.DenseDataset;
import data.Instance;
import data.Instance.InstanceType;
import evaluation.ClassificationEvaluation;

public class MLlib 
{
	
	public static void main(String[] args)
	{
		DenseDataset dataset = new DenseDataset();
		dataset.loadFromSparseFile("train.dat", "test.dat", "", "feature_names.dat", 8);
		//dataset.randomTrainSet(0.8);
		DecisionTreeC45 dt = new DecisionTreeC45();
		dt.buildModel(dataset);
		dt.printTree("tree.xml");
		dt.predict(dataset);
		ClassificationEvaluation.evalPrecision(dataset);
		titanicResult("result.dat", dataset);
	}
	
	public static void titanicResult(String filename, Dataset dataset)
	{
		try 
		{
			FileOutputStream fos = new FileOutputStream(filename);
			PrintStream ps = new PrintStream(fos);
			ps.println("PassengerId,Survived");
			for(Instance inst : dataset.data)
			{
				if(inst.type == InstanceType.Test)
					ps.println(inst.id + "," + (int)inst.predict);
			}
			ps.close();
			fos.close();
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
	
	
}

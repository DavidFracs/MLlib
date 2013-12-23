package main;

import java.io.FileOutputStream;
import java.io.PrintStream;

import classification.bayes.NaiveBayesClassifier;
import classification.bayes.RevisedNaiveBayesClassifier;
import classification.ensemble.AdaBoostClassifier;
import data.Dataset;
import data.Instance;
import data.Instance.InstanceType;
import data.SparseDataset;
import evaluation.ClassificationEvaluation;

public class MLlib 
{
	
	public static void main(String[] args)
	{
		SparseDataset dataset = new SparseDataset();
		dataset.loadFromFile("train_tree.dat", "", "", "feature_names_tree.dat");
		dataset.randomTrainSet(0.8);
		AdaBoostClassifier classifier = new AdaBoostClassifier();
		classifier.buildModel(dataset);
		classifier.predict(dataset);
		ClassificationEvaluation.evalPrecision(dataset);
		
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

package main;

import java.io.FileOutputStream;
import java.io.PrintStream;

import classification.bayes.NaiveBayesClassifier;
import classification.bayes.RevisedNaiveBayesClassifier;
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
		dataset.loadFromFile("train.dat", "", "", "feature_names.dat");
		dataset.randomTrainSet(0.8);
		RevisedNaiveBayesClassifier classifier = new RevisedNaiveBayesClassifier();
		classifier.buildModel(dataset);
		classifier.predict(dataset);
		ClassificationEvaluation.evalPrecision(dataset);
		NaiveBayesClassifier classifier2 = new NaiveBayesClassifier();
		classifier2.buildModel(dataset);
		classifier2.predict(dataset);
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

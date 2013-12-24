package main;

import java.io.FileOutputStream;
import java.io.PrintStream;

import classification.bayes.NaiveBayesClassifier;
import classification.bayes.RevisedNaiveBayesClassifier;
import classification.ensemble.AdaBoostClassifier;
import classification.linear.LogisticRegression;
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
		dataset.scale();
		dataset.fixedFoldTrainSet(3, 1);
		LogisticRegression classifier = new LogisticRegression();
		classifier.buildModel(dataset);
		classifier.predict(dataset);
		ClassificationEvaluation.evalPrecisionAtN(dataset, 0.3);
		ClassificationEvaluation.evalAUC(dataset);
		
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

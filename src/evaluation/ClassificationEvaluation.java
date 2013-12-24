package evaluation;

import java.util.Collections;
import java.util.Comparator;

import data.Dataset;
import data.Instance;
import data.Instance.InstanceType;

public class ClassificationEvaluation 
{
	public static void evalPrecision(Dataset dataset)
	{
		double trainTotal = 0, trainCorrect = 0;
		double testTotal = 0, testCorrect = 0;
		double quizTotal = 0, quizCorrect = 0;
		for(Instance inst : dataset.data)
		{
			if(inst.type == InstanceType.Train)
			{
				trainTotal++;
				if(inst.target == inst.predict)
					trainCorrect++;
			}
			else if(inst.type == InstanceType.Test)
			{
				testTotal++;
				if(inst.target == inst.predict)
					testCorrect++;
			}
			else
			{
				quizTotal++;
				if(inst.target == inst.predict)
					quizCorrect++;
			}
		}
		if(trainTotal > 0)
			System.out.print(String.format("Train: %.4f %d\t", trainCorrect/trainTotal, (int)trainTotal));
		if(testTotal > 0)
			System.out.print(String.format("Test: %.4f %d\t", testCorrect/testTotal, (int)testTotal));
		if(quizTotal > 0)
			System.out.print(String.format("Quiz: %.4f %d\t", quizCorrect/quizTotal, (int)quizTotal));
		System.out.println();
	}
	
	public static void evalPrecisionAtN(Dataset dataset, double per)
	{
		double trainTotal = 0, trainCorrect = 0;
		double testTotal = 0, testCorrect = 0;
		double quizTotal = 0, quizCorrect = 0;
		Collections.sort(dataset.data, new InstanceComparator());
		for(Instance inst : dataset.data)
		{
			if(inst.type == InstanceType.Train && trainTotal < dataset.trainCount * per)
			{
				trainTotal++;
				if(inst.target == 1.0)
					trainCorrect++;
			}
			else if(inst.type == InstanceType.Test && testTotal < dataset.testCount * per)
			{
				testTotal++;
				if(inst.target == 1.0)
					testCorrect++;
			}
			else if(inst.type == InstanceType.Quiz && quizTotal < dataset.quizCount * per)
			{
				quizTotal++;
				if(inst.target == 1.0)
					quizCorrect++;
			}
		}
		if(trainTotal > 0)
			System.out.print(String.format("Train@%d: %.4f\t", (int)trainTotal, trainCorrect/trainTotal));
		if(testTotal > 0)
			System.out.print(String.format("Test@%d: %.4f\t", (int)testTotal, testCorrect/testTotal));
		if(quizTotal > 0)
			System.out.print(String.format("Quiz@%d: %.4f\t",  (int)quizTotal, quizCorrect/quizTotal));
		System.out.println();
	}
	
	public static void evalAUC(Dataset dataset)
	{
		double trainArea = 0, trainPosi = 0, trainRank = dataset.trainCount;
		double testArea = 0, testPosi = 0, testRank = dataset.testCount;
		double quizArea = 0, quizPosi = 0, quizRank = dataset.quizCount;
		Collections.sort(dataset.data, new InstanceComparator());
		for(Instance inst : dataset.data)
		{
			if(inst.type == InstanceType.Train)
			{
				if(inst.target == 1.0)
				{
					trainPosi++;
					trainArea += trainRank;
				}
				trainRank--;	
			}
			else if(inst.type == InstanceType.Test)
			{
				if(inst.target == 1.0)
				{
					testPosi++;
					testArea += testRank;
				}
				testRank--;
			}
			else
			{
				if(inst.target == 1.0)
				{
					quizPosi++;
					quizArea += quizRank;
				}
				quizRank--;
			}
		}
		if(dataset.trainCount > 0)
		{
			double auc = (trainArea - trainPosi * (trainPosi + 1) / 2) / (trainPosi * (dataset.trainCount - trainPosi));
			System.out.print(String.format("Train: %.4f %d\t", auc, dataset.trainCount));	
		}
		if(dataset.testCount > 0)
		{
			double auc = (testArea - testPosi * (testPosi + 1) / 2) / (testPosi * (dataset.testCount - testPosi));
			System.out.print(String.format("Test: %.4f %d\t", auc, dataset.testCount));	
		}
		if(dataset.quizCount > 0)
		{
			double auc = (quizArea - quizPosi * (quizPosi + 1) / 2) / (quizPosi * (dataset.quizCount - quizPosi));
			System.out.print(String.format("Quiz: %.4f %d\t", auc, dataset.quizCount));	
		}
		System.out.println();
	}
}

class InstanceComparator implements Comparator<Instance>
{
	public int compare(Instance s1, Instance s2) 
	{
		double d = s2.predict - s1.predict;
		if(d > 0)
			return 1;
		if(d < 0)
			return -1;
		return 0;
	}
}

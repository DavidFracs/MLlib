package evaluation;

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
}

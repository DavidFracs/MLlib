package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import data.Feature.FeatureType;
import data.Instance.InstanceType;

public class SparseDataset extends Dataset
{
	public SparseDataset()
	{
		
	}

	@Override
	public void loadFromFile(String trainFile, String testFile, String quizFile, String featureFile) 
	{
		System.out.print("Loading dataset... ");
		try 
		{
			if(trainFile != null && trainFile.length() > 0);
				loadDataFromFile(trainFile, InstanceType.Train);
			if(testFile != null && testFile.length() > 0)
				loadDataFromFile(testFile, InstanceType.Test);
			if(quizFile != null && quizFile.length() > 0)
				loadDataFromFile(quizFile, InstanceType.Quiz);
			if(featureFile != null && featureFile.length() > 0)
				loadFeatureFromFile(featureFile);
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		System.out.println(this.toString());
	}

	private void loadFeatureFromFile(String featureFile) throws IOException 
	{
		FileReader fr = new FileReader(featureFile);
		BufferedReader br = new BufferedReader(fr);
		String line = null;
		while((line = br.readLine()) != null)
		{
			String[] tks = line.trim().split("[\t]+");
			int fid = Integer.parseInt(tks[0]);
			int t = Integer.parseInt(tks[1]);
			FeatureType type = t == 0 ? FeatureType.Continuous : FeatureType.Discrete;
			this.feature2Name.put(fid, tks[2]);
			this.feature2Type.put(fid, type);
		}
		br.close();
		fr.close();
	}

	private void loadDataFromFile(String file, InstanceType type) throws IOException 
	{
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line = null;
		while((line = br.readLine()) != null)
		{
			String[] tks = line.trim().split("[\t]+");
			int id = Integer.parseInt(tks[0]);
			double target = Double.parseDouble(tks[1]);
			SparseInstance inst = new SparseInstance(id, target);
			inst.type = type;
			for(int i = 2; i < tks.length; i++)
			{
				int fid = Integer.parseInt(tks[i].split(":")[0]);
				double value = Double.parseDouble(tks[i].split(":")[1]);
				inst.addFeature(fid, value);
				if(fid >= this.featureCount)
					featureCount = fid+1;
			}
			this.addInstance(inst);
		}
		br.close();
		fr.close();
	}
}

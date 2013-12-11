package classification.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import classification.ClassificationAlgorithm;
import data.Dataset;
import data.Feature.FeatureType;
import data.Instance;

class InstanceComparator implements Comparator<Instance>
{
	public int fid = -1;
	
	public void setFid(int fid)
	{
		this.fid = fid;
	}
	
	public int compare(Instance s1, Instance s2) 
	{
		if(!s1.containsFeature(fid)) return 1;
		if(!s2.containsFeature(fid)) return -1;
		
		if(s1.getFeature(fid) - s2.getFeature(fid) > 0)
			return 1;
		if(s1.getFeature(fid) - s2.getFeature(fid) < 0)
			return -1;
		return 0;
		 
	}
}

class TreeNodeC45
{
	double result = -1;
	double value = 0;
	double support = 0;
	double errorCount = 0;
	FeatureType featureType = null;
	HashMap<Double, TreeNodeC45> children = null;
}

public class DecisionTreeC45 implements ClassificationAlgorithm
{
	private TreeNodeC45 root = null;
	private HashSet<Integer> featureUsed = null;
	private InstanceComparator instanceComparator = new InstanceComparator();
	
	public DecisionTreeC45()
	{
		
	}
	
	private TreeNodeC45 buildTree(Dataset dataset, ArrayList<Instance> data)
	{
		double  posiCount = 0, negaCount = 0, totalCount = 0;
		double curEn = calculateEntropy(negaCount, posiCount, totalCount);
		//get feature of max gain rate
		int maxGRFid = -1;
		double maxGR = 0;
		for(int fid = 0; fid < dataset.featureCount; fid++)
		{
			if(featureUsed.contains(fid)) continue;
			double curGR = getGainRate(data, fid, dataset.getFeatureType(fid), curEn);
			if(curGR > maxGR)
			{
				maxGRFid = fid;
				maxGR = curGR;
			}
		}
		featureUsed.add(maxGRFid);
		//split with maxGRFid
		ArrayList<ArrayList<Instance>> dataSplits = new ArrayList<ArrayList<Instance>>();
		double thres = splitDataset(dataSplits, data, maxGRFid, dataset.getFeatureType(maxGRFid));
		
		
		featureUsed.remove(maxGRFid);
		return null;
	}
	
	
	private double splitDataset(ArrayList<ArrayList<Instance>> dataSplits, ArrayList<Instance> data, int maxGRFid, FeatureType type) 
	{
		if(type == FeatureType.Continuous)
		{
			return splitDatasetContinuous(dataSplits, data, maxGRFid);
		}
		else
		{
			return splitDatasetDiscrete(dataSplits, data, maxGRFid);
		}
	}

	private double splitDatasetDiscrete(ArrayList<ArrayList<Instance>> dataSplits, ArrayList<Instance> data, int maxGRFid) 
	{
		HashMap<Double, ArrayList<Instance>> splits = new HashMap<Double, ArrayList<Instance>>();
		for(int i = 0; i < data.size(); i++)
		{
			if(!data.get(i).containsFeature(maxGRFid)) continue;
			double val = data.get(i).getFeature(maxGRFid);
			if(!splits.containsKey(val)) splits.put(val, new ArrayList<Instance>());
			splits.get(val).add(data.get(i));
		}
		for(double val : splits.keySet())
		dataSplits.add(splits.get(val));
		return 0;
	}

	private double splitDatasetContinuous(ArrayList<ArrayList<Instance>> dataSplits, ArrayList<Instance> data, int maxGRFid) 
	{
		//stats
		double fPosiCount = 0, fNegaCount = 0, bPosiCount = 0, bNegaCount = 0;
		double fCount = 0, bCount = 0, validCount = 0;
		instanceComparator.fid = maxGRFid;
		Collections.sort(data, instanceComparator);
		for(int i = 0; i < data.size(); i++)
		{
			if(!data.get(i).containsFeature(maxGRFid)) break;
			if(data.get(i).target == 0)
				bNegaCount++;
			bCount++;
		}
		bPosiCount = bCount - bNegaCount;
		validCount = bCount;
		// max Gain
		double minEn = Double.MAX_VALUE;
		double thres = 0;
		for(int i = 0; i < validCount - 1; i++)
		{
			fCount++; bCount--;
			if(data.get(i).target == 0)
			{
				fNegaCount++; bNegaCount--;
			}
			else
			{
				fPosiCount++; bNegaCount--;
			}
			double en = fCount / validCount * calculateEntropy(fPosiCount, fNegaCount, fCount) + 
					fCount / validCount * calculateEntropy(fPosiCount, fNegaCount, fCount); 
			if(en < minEn)
			{
				minEn = en;
				thres = data.get(i).getFeature(maxGRFid);
			}
		}
		//split
		dataSplits.add(new ArrayList<Instance>());
		dataSplits.add(new ArrayList<Instance>());
		for(int i = 0; i < validCount; i++)
		{
			if(data.get(i).getFeature(maxGRFid) <= thres)
				dataSplits.get(0).add(data.get(i));
			else
				dataSplits.get(1).add(data.get(i));
		}
		return thres;
	}

	public void buildModel(Dataset dataset) 
	{
		ArrayList<Instance> data = new ArrayList<Instance>();
		data.addAll(dataset.data);
		root = buildTree(dataset, data);
		
	}
	
	private double calculateEntropy(double a, double b, double total)
	{
		if (a == 0 || b == 0) return 0;
		return 0 - a/total * Math.log(a/total) - b/total * Math.log(b/total);
	}
	
	private double getGainRateContinuous(ArrayList<Instance> data, int fid, double curEn)
	{
		//statistics
		double fPosiCount = 0, fNegaCount = 0, bPosiCount = 0, bNegaCount = 0;
		double fCount = 0, bCount = 0, validCount = 0, totalCount = data.size();
		instanceComparator.fid = fid;
		Collections.sort(data, instanceComparator);
		for(int i = 0; i < data.size(); i++)
		{
			if(!data.get(i).containsFeature(fid)) break;
			if(data.get(i).target == 0)
				bNegaCount++;
			bCount++;
		}
		bPosiCount = bCount - bNegaCount;
		validCount = bCount;
		// averageGain
		double totalGain = 0;
		for(int i = 0; i < validCount - 1; i++)
		{
			fCount++; bCount--;
			if(data.get(i).target == 0)
			{
				fNegaCount++; bNegaCount--;
			}
			else
			{
				fPosiCount++; bNegaCount--;
			}
			double en = fCount / validCount * calculateEntropy(fPosiCount, fNegaCount, fCount) + 
					fCount / validCount * calculateEntropy(fPosiCount, fNegaCount, fCount); 
			totalGain = totalGain + curEn - en;
		}
		double averGain = totalGain / (validCount - 1);
		
		//maxGainRate
		bCount = validCount; fCount = 0;
		bPosiCount += fPosiCount; fPosiCount = 0;
		bNegaCount += fNegaCount; fNegaCount = 0; 
		double maxGR = 0;
		for(int i = 0; i < validCount - 1; i++)
		{
			fCount++; bCount--;
			if(data.get(i).target == 0)
			{
				fNegaCount++; bNegaCount--;
			}
			else
			{
				fPosiCount++; bNegaCount--;
			}
			double en = fCount / validCount * calculateEntropy(fPosiCount, fNegaCount, fCount) + 
					fCount / validCount * calculateEntropy(fPosiCount, fNegaCount, fCount); 
			double gain = curEn - en;
			if(gain < averGain) continue;
			double splitEn = calculateEntropy(fCount, bCount, validCount);
			double gRate = gain / splitEn;
			if(gRate > maxGR) maxGR = gRate;
		}
		return maxGR * validCount / totalCount;
	}
	
	private double getGainRateDiscrete(ArrayList<Instance> data, int fid, double curEn)
	{
		HashMap<Double, double[]> stat = new HashMap<Double, double[]>();
		double validCount = 0;
		for(int i = 0; i < data.size(); i++)
		{
			if(!data.get(i).containsFeature(fid)) continue;
			validCount++;
			double val = data.get(i).getFeature(fid);
			if(!stat.containsKey(val))
				stat.put(val, new double[3]);
			stat.get(val)[(int)(data.get(i).target)]++;
			stat.get(val)[2]++;
		}
		double en = 0;
		double splitEn = 0;
		for(Map.Entry<Double, double[]> pair : stat.entrySet())
		{
			double[] count = pair.getValue();
			en = en + count[2]/validCount * calculateEntropy(count[0], count[1], count[2]);
			splitEn = splitEn + count[2]/validCount * Math.log(count[2]/validCount);
		}
		return (curEn - en) / splitEn * validCount / data.size() ;
	}
	
	private double getGainRate(ArrayList<Instance> data, int fid, FeatureType type, double curEn)
	{
		if(type == FeatureType.Continuous)
			return getGainRateContinuous(data, fid, curEn);
		else
			return getGainRateDiscrete(data, fid, curEn);
	}
}

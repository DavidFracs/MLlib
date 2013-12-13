package classification.tree;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import classification.ClassificationAlgorithm;
import data.Dataset;
import data.Feature.FeatureType;
import data.Instance;
import data.Instance.InstanceType;

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
	public boolean isLeaf = false;
	
	public double result = -1;
	public double support = 0;
	
	public double value = 0;
	public int fid = -1;
	public double errorCount = 0;
	public FeatureType featureType = null;
	public HashMap<Double, TreeNodeC45> children = new HashMap<Double, TreeNodeC45>();
	
	public TreeNodeC45(double value, int fid, FeatureType type, double result, double support, double errorCount)
	{
		this.value = value;
		this.featureType = type;
		this.fid = fid;
		this.result = result;
		this.support = support;
		this.errorCount = errorCount;
		this.isLeaf = false;
	}
	
	public TreeNodeC45(double result,  double support, double errorCount)
	{
		this.result = result;
		this.support  = support;
		this.errorCount = errorCount;
		this.isLeaf = true;
	}
}

public class DecisionTreeC45 implements ClassificationAlgorithm
{
	private TreeNodeC45 root = null;
	private HashSet<Integer> featureUsed = null;
	private InstanceComparator instanceComparator = new InstanceComparator();
	
	private int MinNodeSize = 1;
	private double ConfidenceLevel = 1;
	private double ExtraErrorCount = 0.5;
	
	public DecisionTreeC45()
	{
		featureUsed = new HashSet<Integer>();
	}
	
	public void buildModel(Dataset dataset) 
	{
		ArrayList<Instance> data = new ArrayList<Instance>();
		for(Instance inst : dataset.data)
			if(inst.type == InstanceType.Train)
				data.add(inst);
		root = buildTree(dataset, data);
		prune();
	}
	
	public void prune()
	{
		double[] errorAdded = new double[1];
		prune(root, errorAdded);
	}
	
	private double prune(TreeNodeC45 curRoot, double errorAdded[])
	{
		if(curRoot.isLeaf) 
		{
			double curError = curRoot.support * estimateErrorRate(curRoot.support + ExtraErrorCount, curRoot.errorCount + ExtraErrorCount);
			errorAdded[0] += ExtraErrorCount;
			return curError;
		}
		double childError = 0;
		for(double val : curRoot.children.keySet())
		{
			double[] nextAdded = new double[1];
			childError += prune(curRoot.children.get(val), nextAdded);
			errorAdded[0] += nextAdded[0];
		}
		double curError = curRoot.support * estimateErrorRate(curRoot.support + errorAdded[0], curRoot.errorCount + errorAdded[0]);
		if(curError < childError)
		{
			curError = curRoot.support * estimateErrorRate(curRoot.support + ExtraErrorCount, curRoot.errorCount + ExtraErrorCount);
			errorAdded[0] = ExtraErrorCount;
			curRoot.isLeaf = true;
			curRoot.children.clear();
			return curError;
		}
		return childError; 
	}
	
	
	public void predict(Dataset dataset)
	{
		for(Instance inst : dataset.data)
			predict(inst);
	}
	
	public void predict(Instance inst)
	{
		int count[] = new int[2];
		predict(inst, root, count);	
		if(count[0] >= count[1])
			inst.predict = 0.0;
		else
			inst.predict = 1.0;
	}
	
	private void predict(Instance inst, TreeNodeC45 curRoot, int[] count) 
	{
		if(curRoot.isLeaf)
		{
			if(curRoot.result == 0)
			{
				count[0] += curRoot.support - curRoot.errorCount;
				count[1] += curRoot.errorCount;
			}
			else
			{
				count[1] += curRoot.support - curRoot.errorCount;
				count[0] += curRoot.errorCount;
			}
			return;
		}
		if(inst.containsFeature(curRoot.fid))
		{
			double value = inst.getFeature(curRoot.fid);
			if(curRoot.featureType == FeatureType.Continuous)
			{
				if(value <= curRoot.value)
					predict(inst, curRoot.children.get(-1.0), count);
				else
					predict(inst, curRoot.children.get(1.0), count);
				return;
			}
			else if(curRoot.children.containsKey(value))
			{
				predict(inst, curRoot.children.get(value), count);
				return;
			}
		}
		//no this value or no this feature
		for(double val : curRoot.children.keySet())
			predict(inst, curRoot.children.get(val), count);
	}

	private TreeNodeC45 buildTree(Dataset dataset, ArrayList<Instance> data)
	{
		//check
		double  posiCount = 0, negaCount = 0, totalCount = 0;
		for(int i = 0; i < data.size(); i++)
		{
			totalCount += data.get(i).weight;
			if(data.get(i).target == 1) posiCount += data.get(i).weight;
			else negaCount += data.get(i).weight;
		}
		double result = posiCount > negaCount ? 1 : 0;
		double error = result == 1 ? negaCount : posiCount;
		if(totalCount < MinNodeSize || error == 0 || featureUsed.size() == dataset.featureCount)
			return new TreeNodeC45(result, totalCount, error);
		
		//get feature of max gain rate
		double curEn = calculateEntropy(negaCount, posiCount, totalCount);
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
		if(maxGR == 0)
			return new TreeNodeC45(result, totalCount, error);
		featureUsed.add(maxGRFid);
		//split with maxGRFid
		ArrayList<ArrayList<Instance>> dataSplits = new ArrayList<ArrayList<Instance>>();
		ArrayList<Double> valueSplits = new ArrayList<Double>();
		double thres = splitDataset(dataSplits, valueSplits, data, maxGRFid, dataset.getFeatureType(maxGRFid));
		data.clear();
		//build tree node 
		TreeNodeC45 ret = new TreeNodeC45(thres, maxGRFid, dataset.getFeatureType(maxGRFid), result,  totalCount, error);
		for(int i = 0; i < dataSplits.size(); i++)
			ret.children.put(valueSplits.get(i),  buildTree(dataset, dataSplits.get(i)));
		featureUsed.remove(maxGRFid);
		return ret;
	}
	
	
	private double splitDataset(ArrayList<ArrayList<Instance>> dataSplits, ArrayList<Double> valueSplits, ArrayList<Instance> data, int maxGRFid, FeatureType type) 
	{
		if(type == FeatureType.Continuous)
		{
			return splitDatasetContinuous(dataSplits, valueSplits, data, maxGRFid);
		}
		else
		{
			return splitDatasetDiscrete(dataSplits, valueSplits, data, maxGRFid);
		}
	}

	private double splitDatasetDiscrete(ArrayList<ArrayList<Instance>> dataSplits, ArrayList<Double> valueSplits, ArrayList<Instance> data, int maxGRFid) 
	{
		HashMap<Double, ArrayList<Instance>> splits = new HashMap<Double, ArrayList<Instance>>();
		ArrayList<Instance> missInstances = new ArrayList<Instance>();
		double sum = 0;
		HashMap<Double, Double> splitSum = new HashMap<Double, Double>(); 
		for(int i = 0; i < data.size(); i++)
		{
			if(!data.get(i).containsFeature(maxGRFid))
			{
				missInstances.add(data.get(i));
				continue;
			}
			double val = data.get(i).getFeature(maxGRFid);
			sum += data.get(i).weight;
			if(!splits.containsKey(val)) 
			{
				splits.put(val, new ArrayList<Instance>());
				splitSum.put(val, 0.0);
			}
			splits.get(val).add(data.get(i));
			splitSum.put(val, data.get(i).weight + splitSum.get(val));
		}
		for(double val : splits.keySet())
		{
			double weight = splitSum.get(val) / sum;
			for(Instance inst : missInstances)
			{
				Instance pseudoInst = inst.clone();
				pseudoInst.weight = pseudoInst.weight * weight;
				splits.get(val).add(pseudoInst);
			}	
		}
		for(double val : splits.keySet())
		{
			dataSplits.add(splits.get(val));
			valueSplits.add(val);
		}
		return 0;
	}

	private double splitDatasetContinuous(ArrayList<ArrayList<Instance>> dataSplits, ArrayList<Double> valueSplits, ArrayList<Instance> data, int maxGRFid) 
	{
		//statistics
		double fPosiCount = 0, fNegaCount = 0, bPosiCount = 0, bNegaCount = 0;
		double fCount = 0, bCount = 0, validCount = 0;
		instanceComparator.fid = maxGRFid;
		Collections.sort(data, instanceComparator);
		for(int i = 0; i < data.size(); i++)
		{
			if(!data.get(i).containsFeature(maxGRFid)) break;
			if(data.get(i).target == 0)
				bNegaCount += data.get(i).weight;
			bCount += data.get(i).weight;
			validCount++;
		}
		bPosiCount = bCount - bNegaCount;
		// max Gain
		double minEn = Double.MAX_VALUE;
		double thres = 0;
		for(int i = 0; i < validCount - 1; i++)
		{
			fCount += data.get(i).weight; 
			bCount -= data.get(i).weight;
			if(data.get(i).target == 0)
			{
				fNegaCount += data.get(i).weight; 
				bNegaCount -= data.get(i).weight;
			}
			else
			{
				fPosiCount += data.get(i).weight; 
				bPosiCount -= data.get(i).weight;
			}
			if(data.get(i).getFeature(maxGRFid) == data.get(i+1).getFeature(maxGRFid)) continue;
			double en = bCount / (fCount + bCount) * calculateEntropy(bPosiCount, bNegaCount, bCount) + 
					fCount / (fCount + bCount) * calculateEntropy(fPosiCount, fNegaCount, fCount); 
			if(en < minEn)
			{
				minEn = en;
				thres = data.get(i).getFeature(maxGRFid);
			}
		}
		//split
		dataSplits.add(new ArrayList<Instance>());
		dataSplits.add(new ArrayList<Instance>());
		valueSplits.add(-1.0);
		valueSplits.add(1.0);
		double[] weightSum = new double[2];
		for(int i = 0; i < validCount; i++)
		{
			if(data.get(i).getFeature(maxGRFid) <= thres)
			{
				dataSplits.get(0).add(data.get(i));
				weightSum[0] += data.get(i).weight;
			}
			else
			{
				dataSplits.get(1).add(data.get(i));
				weightSum[1] += data.get(i).weight;
			}
		}
		double weight = weightSum[0] / (weightSum[0] + weightSum[1]);
		for(int i = (int)validCount; i < data.size(); i++)
		{
			Instance pseudoInst = data.get(i).clone();
			pseudoInst.weight = pseudoInst.weight * weight;
			dataSplits.get(0).add(pseudoInst);
			pseudoInst = data.get(i).clone();
			pseudoInst.weight = pseudoInst.weight * (1 - weight);
			dataSplits.get(1).add(pseudoInst);
		}
		return thres;
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
				bNegaCount += data.get(i).weight;
			bCount += data.get(i).weight;
			validCount ++;
		}
		bPosiCount = bCount - bNegaCount;
		// averageGain
		double totalGain = 0;
		
		for(int i = 0; i < validCount - 1; i++)
		{
			fCount += data.get(i).weight; 
			bCount -= data.get(i).weight;
			if(data.get(i).target == 0)
			{
				fNegaCount += data.get(i).weight; 
				bNegaCount -= data.get(i).weight;
				
			}
			else
			{
				fPosiCount += data.get(i).weight; 
				bPosiCount -= data.get(i).weight;
				
			}
			double en = bCount / (bCount + fCount) * calculateEntropy(bPosiCount, bNegaCount, bCount) + 
					fCount / (bCount + fCount) * calculateEntropy(fPosiCount, fNegaCount, fCount); 
			totalGain = totalGain + curEn - en;
		}
		double averGain = totalGain / (validCount - 1);
		
		//maxGainRate
		bCount = bCount + fCount; fCount = 0;
		bPosiCount += fPosiCount; fPosiCount = 0;
		bNegaCount += fNegaCount; fNegaCount = 0; 
		double maxGR = 0;
		for(int i = 0; i < validCount - 1; i++)
		{
			fCount += data.get(i).weight; 
			bCount -= data.get(i).weight;
			if(data.get(i).target == 0)
			{
				fNegaCount += data.get(i).weight; 
				bNegaCount -= data.get(i).weight;
			}
			else
			{
				fPosiCount += data.get(i).weight; 
				bPosiCount -= data.get(i).weight;
			}
			if(data.get(i).getFeature(fid) == data.get(i+1).getFeature(fid)) continue;
			double en = fCount / (bCount + fCount) * calculateEntropy(fPosiCount, fNegaCount, fCount) + 
					fCount / (bCount + fCount) * calculateEntropy(fPosiCount, fNegaCount, fCount); 
			double gain = curEn - en;
			if(gain < averGain) continue;
			double splitEn = calculateEntropy(fCount, bCount, (bCount + fCount));
			double gRate = gain / splitEn;
			if(gRate > maxGR) maxGR = gRate;
		}
		return maxGR * (bCount + fCount) / totalCount;
	}
	
	private double getGainRateDiscrete(ArrayList<Instance> data, int fid, double curEn)
	{
		HashMap<Double, double[]> stat = new HashMap<Double, double[]>();
		double validCount = 0;
		for(int i = 0; i < data.size(); i++)
		{
			if(!data.get(i).containsFeature(fid)) continue;
			validCount += data.get(i).weight;
			double val = data.get(i).getFeature(fid);
			if(!stat.containsKey(val))
				stat.put(val, new double[3]);
			stat.get(val)[(int)(data.get(i).target)] += data.get(i).weight;
			stat.get(val)[2] += data.get(i).weight;
		}
		double en = 0;
		double splitEn = 0;
		for(Map.Entry<Double, double[]> pair : stat.entrySet())
		{
			double[] count = pair.getValue();
			en = en + count[2]/validCount * calculateEntropy(count[0], count[1], count[2]);
			splitEn = splitEn - count[2]/validCount * Math.log(count[2]/validCount);
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
	
	public void printTree(String treeDumpFile)
	{
		try 
		{
			FileOutputStream fos = new FileOutputStream(treeDumpFile);
			PrintStream ps = new PrintStream(fos);
			printTree(root, ps);
			ps.close();
			fos.close();
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}

	private void printTree(TreeNodeC45 curRoot, PrintStream ps) 
	{
		ps.println("<TreeNode>");
		if(curRoot.isLeaf)
		{
			ps.println("<result>" + curRoot.result + "</result>");
			ps.println("<support>" + curRoot.support + "</support>");
			ps.println("<error>" + curRoot.errorCount + "</error>");
		}
		else
		{
			ps.println("<fid>" + curRoot.fid + "</fid>");
			ps.println("<type>" + curRoot.featureType + "</type>");
			ps.println("<value>" + curRoot.value + "</value>");
			ps.println("<support>" + curRoot.support + "</support>");
			ps.println("<children>");
			for(double val : curRoot.children.keySet())
				printTree(curRoot.children.get(val), ps);
			ps.println("</children>");
		}
		ps.println("</TreeNode>");
	}
	
	private double estimateErrorRate(double support, double error)
	{
		double p = (error) / (support);
		return p + ConfidenceLevel * Math.sqrt(p * (1 - p) / (support));
	}
}

package data;

import java.util.HashMap;

public class SparseInstance extends Instance 
{
	public HashMap<Integer, Double> features = new HashMap<Integer, Double>();
	
	public SparseInstance() {}
	
	public SparseInstance(int id)
	{
		this.id = id;
	}
	
	public SparseInstance(int id, double target)
	{
		this.id = id;
		this.target = target;
	}
	
	public void addFeature(int fid, double fv)
	{
		this.features.put(fid, fv);
	}
	
	public void setFeature(int fid, double fv)
	{
		this.features.put(fid, fv);
	}
	
	public void addFeature(HashMap<Integer, Double> featuresToAdd)
	{
		this.features.putAll(featuresToAdd);	
	}
	
	public void removeFeature(int fid)
	{
		this.features.remove(fid);
	}
	
	public void removeAllFeatures()
	{
		this.features.clear();
	}
	
	public double getFeature(int fid)
	{
		if(features.containsKey(fid))
			return features.get(fid);
		return Double.NaN;
	}
	
	public boolean containsFeature(int fid)
	{
		return this.features.containsKey(fid);
	}
	
	public Instance clone() {
		SparseInstance newInst = new SparseInstance(this.id, this.target);
		newInst.type = this.type;
		newInst.weight = this.weight;
		newInst.predict = this.predict;
		newInst.features = this.features;
		return newInst;
	}

	public int[] getFeatureIds() 
	{
		int [] fids = new int[features.size()];
		int n = 0;
		for(int fid : features.keySet())
			fids[n++] = fid;
		return fids;
	}
}

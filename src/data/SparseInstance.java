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
}

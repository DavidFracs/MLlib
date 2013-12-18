package data;

import java.util.ArrayList;

public class DenseInstance extends Instance
{
	public ArrayList<Double> features = new ArrayList<Double>();
	
	public DenseInstance() {}
	
	public DenseInstance(int id)
	{
		this.id = id;
	}
	
	public DenseInstance(int id, double target)
	{
		this.id = id;
		this.target = target;
	}
	
	public void addFeature(int fid, double fv)
	{
		this.features.set(fid, fv);
	}
	
	public void setFeature(int fid, double fv)
	{
		if(features.size() > fid)
			this.features.set(fid, fv);
		else System.err.println(String.format("DenseIntance %d doesn't contain feature %d", this.id, fid));
	}
	
	public void addFeature(ArrayList<Double> featuresToAdd)
	{
		this.features.clear();
		this.features.addAll(featuresToAdd);	
	}
	
	public void removeFeature(int fid)
	{
		System.err.println("DenseIntance dosn't support removing features ... Maybe set it to zero will work");
	}
	
	public void removeAllFeatures()
	{
		System.err.println("DenseIntance dosn't support removing features ... Maybe set it to zero will work");
	}
	
	public double getFeature(int fid)
	{
		if(features.size() > fid)
			return features.get(fid);
		return Double.NaN;
	}
	
	public boolean containsFeature(int fid)
	{
		return this.features.size() > fid;
	}

	public Instance clone() {
		DenseInstance newInst = new DenseInstance(this.id, this.target);
		newInst.type = this.type;
		newInst.weight = this.weight;
		newInst.predict = this.predict;
		newInst.features = this.features;
		return newInst;
	}
	
	public int[] getFeatureIds() 
	{
		int [] fids = new int[features.size()];
		for(int i = 0; i < features.size(); i++)
			fids[i] = i;
		return fids;
	}
}

package data;

public abstract class Instance 
{
	public static enum InstanceType {Train, Test, Quiz};
	
	public int id = -1;
	public double target = -1;
	public double predict = -2;
	public InstanceType type = InstanceType.Train;
	
	public abstract void addFeature(int fid, double fv);
	public abstract void setFeature(int fid, double fv);
	public abstract void removeFeature(int fid);
	public abstract void removeAllFeatures();
	public abstract double getFeature(int fid);
	public abstract boolean containsFeature(int fid);
}

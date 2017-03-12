package net.mnementh64.neural.model.weight;

public class UnitFunction extends WeightInitFunction
{

	public UnitFunction()
	{
		super(Type.UNIT);
	}

	@Override
	public double[][] init(int s1, int s2)
	{
		double[][] weights = new double[s1][s2];
		for (int i1 = 0; i1 < s1; i1++)
			for (int i2 = 0; i2 < s2; i2++)
				weights[i1][i2] = 1.0;
		return weights;
	}
}

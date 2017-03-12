package net.mnementh64.neural.model.weight;

import java.util.Random;

public class RandomUniformFunction extends WeightInitFunction
{

	double min;
	double max;

	public RandomUniformFunction(double min, double max)
	{
		this.min = min;
		this.max = max;
	}

	@Override
	public double[][] init(int s1, int s2)
	{
		double[][] weights = new double[s1][s2];
		Random random = new Random();
		random.setSeed(System.currentTimeMillis());
		for (int i1 = 0; i1 < s1; i1++)
			for (int i2 = 0; i2 < s2; i2++)
				weights[i1][i2] = (max - min) * random.nextDouble() + min;

		return weights;
	}
}

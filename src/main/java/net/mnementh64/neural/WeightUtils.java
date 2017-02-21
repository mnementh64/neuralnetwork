package net.mnementh64.neural;

import java.util.Random;

import net.mnementh64.neural.layer.WeightInitFunction;

/**
 * Created by scaillet on 2/21/17.
 */
public class WeightUtils
{

	public static float[][] init(int s1, int s2, WeightInitFunction weightInitFunction)
	{
		Random random = new Random();

		float[][] weights = new float[s1][s2];
		for (int i1 = 0; i1 < s1; i1++)
			for (int i2 = 0; i2 < s2; i2++)
				weights[i1][i1] = 2.0f * (random.nextFloat() - 0.5f);

		return new float[s1][s2];
	}
}

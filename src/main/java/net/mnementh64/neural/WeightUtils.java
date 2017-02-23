package net.mnementh64.neural;

import java.util.Random;

import net.mnementh64.neural.layer.WeightInitFunction;

public class WeightUtils
{

	public static float[][] init(int s1, int s2, WeightInitFunction weightInitFunction)
	{
		float[][] weights = new float[s1][s2];

		switch (weightInitFunction)
		{
			case RANDOM:
				Random random = new Random();
				for (int i1 = 0; i1 < s1; i1++)
					for (int i2 = 0; i2 < s2; i2++)
						weights[i1][i2] = 2.0f * (random.nextFloat() - 0.5f);
				break;
			case UNIT:
				for (int i1 = 0; i1 < s1; i1++)
					for (int i2 = 0; i2 < s2; i2++)
						weights[i1][i2] = 1.0f;
				break;
		}

		return weights;
	}
}

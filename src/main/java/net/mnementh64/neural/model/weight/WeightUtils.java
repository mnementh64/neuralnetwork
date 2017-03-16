package net.mnementh64.neural.model.weight;

public class WeightUtils
{

	public static final WeightInitFunction unitFunction = new UnitFunction(1.0);
	public static final WeightInitFunction normalizedFunction = new RandomUniformFunction(-1.0, 1.0);
	public static final WeightInitFunction gaussianNormalizedFunction = new RandomGaussianFunction(-1.0, 1.0);
}

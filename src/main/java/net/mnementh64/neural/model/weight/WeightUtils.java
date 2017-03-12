package net.mnementh64.neural.model.weight;

public class WeightUtils
{

	public static final WeightInitFunction unitFunction = new UnitFunction();
	public static final WeightInitFunction normalizedFunction = new RandomUniformFunction(-1.0, 1.0);
}

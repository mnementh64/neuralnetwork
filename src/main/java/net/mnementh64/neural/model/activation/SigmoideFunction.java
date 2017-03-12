package net.mnementh64.neural.model.activation;

public class SigmoideFunction extends ActivationFunction
{

	public SigmoideFunction()
	{
		super(Type.SIGMOID);
	}

	@Override
	public double apply(double x)
	{
		return 1.0 / (1.0 + Math.exp(-x));
	}

	@Override
	public double applyDerivative(double x)
	{
		return apply(x) * (1.0f - apply(x));
	}
}

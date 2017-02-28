package net.mnementh64.neural.layer;

public class HiddenLayer extends Layer
{

	public HiddenLayer() {}

	public HiddenLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(activationFunction == null ? ActivationFunction.SIGMOID : activationFunction, nbNodes);
	}
}

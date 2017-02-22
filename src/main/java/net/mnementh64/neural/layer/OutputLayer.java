package net.mnementh64.neural.layer;

public class OutputLayer extends Layer
{

	public OutputLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(activationFunction == null ? ActivationFunction.SIGMOID : activationFunction, nbNodes);
	}
}

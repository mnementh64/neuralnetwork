package net.mnementh64.neural.model.layer;

import net.mnementh64.neural.model.ActivationFunction;

public class HiddenLayer extends Layer
{

	public HiddenLayer() {}

	public HiddenLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(activationFunction == null ? ActivationFunction.SIGMOID : activationFunction, nbNodes);
	}
}

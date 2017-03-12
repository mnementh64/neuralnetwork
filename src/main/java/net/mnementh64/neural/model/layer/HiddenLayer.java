package net.mnementh64.neural.model.layer;

import net.mnementh64.neural.model.activation.ActivationFunction;
import net.mnementh64.neural.model.activation.ActivationUtils;

public class HiddenLayer extends Layer
{

	public HiddenLayer()
	{
		super(Type.HIDDEN);
	}

	public HiddenLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(Type.HIDDEN, activationFunction == null ? ActivationUtils.sigmoid : activationFunction, nbNodes);
	}
}

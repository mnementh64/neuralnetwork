package net.mnementh64.neural.layer;

public class InputLayer extends Layer
{

	public InputLayer()
	{
		super();
	}

	public InputLayer(int nbNodes)
	{
		super(null, nbNodes);
	}

	@Override
	public void feedForward(Layer previousLayer) throws Exception
	{
		// do nothing for first layer
	}

}

package net.mnementh64.neural.model.layer;

import java.util.List;
import java.util.stream.IntStream;

import net.mnementh64.neural.model.ActivationFunction;

public class OutputLayer extends Layer
{

	public OutputLayer()
	{
	}

	public OutputLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(activationFunction == null ? ActivationFunction.SIGMOID : activationFunction, nbNodes);
	}

	public void computeError(List<Double> expectedValues) throws Exception
	{
		if (expectedValues.size() != getNbNodes())
			throw new Exception("Expected values are bad sized for the output layer : get " + expectedValues.size() + " items and expected " + getNbNodes());

		IntStream.range(0, expectedValues.size())
				.forEach(i -> nodes.get(i).delta = expectedValues.get(i) - nodes.get(i).value);
	}
}

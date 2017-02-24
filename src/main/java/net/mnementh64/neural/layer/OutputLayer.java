package net.mnementh64.neural.layer;

import java.util.List;
import java.util.stream.IntStream;

public class OutputLayer extends Layer
{

	public OutputLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(activationFunction == null ? ActivationFunction.SIGMOID : activationFunction, nbNodes);
	}

	public void computeError(List<Float> expectedValues) throws Exception
	{
		if (expectedValues.size() != size())
			throw new Exception("Expected values are bad sized for the output layer : get " + expectedValues.size() + " items and expected " + size());

		IntStream.range(0, expectedValues.size())
				.forEach(i -> nodes.get(i).delta = expectedValues.get(i) - nodes.get(i).value);
	}
}

package net.mnementh64.neural;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import net.mnementh64.neural.layer.ActivationFunction;
import net.mnementh64.neural.layer.WeightInitFunction;

public class NetworkTest
{

	@Test
	public void networkTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightInitFunction.RANDOM)
				.addLayer(2)
				.addLayer(5, ActivationFunction.SIGMOID)
				.addLayer(1, ActivationFunction.SIGMOID)
				.build();

		List<Float> input = Arrays.asList(1.2f, -3.4f);
		List<Float> output = network.feedForward(input);

		Assert.assertNotNull(output);
		Assert.assertTrue(output.size() == 1);
	}
}

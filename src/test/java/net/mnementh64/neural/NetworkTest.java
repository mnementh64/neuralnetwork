package net.mnementh64.neural;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import net.mnementh64.neural.layer.ActivationFunction;
import net.mnementh64.neural.layer.WeightInitFunction;

public class NetworkTest
{

	@Test
	public void networkBuildTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightInitFunction.UNIT)
				.addLayer(2)
				.addLayer(5, ActivationFunction.SIGMOID)
				.addLayer(1, ActivationFunction.SIGMOID)
				.build();

		Assert.assertTrue(network.size() == 3);

		Assert.assertTrue(network.getLayerSize(0) == 2);
		Assert.assertTrue(network.getLayerSize(1) == 5);
		Assert.assertTrue(network.getLayerSize(2) == 1);
	}

	@Test(expected = Exception.class)
	public void networkRunTestBadInputSize() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightInitFunction.UNIT)
				.addLayer(2)
				.addLayer(3, ActivationFunction.SIGMOID)
				.addLayer(1, ActivationFunction.SIGMOID)
				.build();

		List<Float> input = Collections.singletonList(3f);
		network.feedForward(input);
	}

	@Test
	public void networkFeedForwardTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightInitFunction.UNIT)
				.addLayer(2)
				.addLayer(3, ActivationFunction.IDENTITY)
				.addLayer(1, ActivationFunction.IDENTITY)
				.build();

		List<Float> input = Arrays.asList(2f, 3f);
		List<Float> output = network.feedForward(input);

		Assert.assertNotNull(output);
		Assert.assertTrue(output.size() == 1);
		Assert.assertTrue(output.get(0) == 15);
	}

	@Test
	public void networkComplexFeedForwardTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightInitFunction.UNIT)
				.addLayer(10)
				.addLayer(32, ActivationFunction.SIGMOID)
				.addLayer(24, ActivationFunction.SIGMOID)
				.addLayer(124, ActivationFunction.SIGMOID)
				.addLayer(234, ActivationFunction.SIGMOID)
				.addLayer(23, ActivationFunction.SIGMOID)
				.build();

		List<Float> input = Arrays.asList(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f);
		List<Float> output = network.feedForward(input);

		Assert.assertNotNull(output);
		Assert.assertTrue(output.size() == 23);
	}

	@Test(expected = Exception.class)
	public void networkRetroPropagateTestBadSize() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightInitFunction.UNIT)
				.addLayer(2)
				.addLayer(3, ActivationFunction.SIGMOID)
				.addLayer(1, ActivationFunction.SIGMOID)
				.build();

		List<Float> input = Arrays.asList(2f, 3f);
		network.feedForward(input);
		List<Float> expectedValues = Arrays.asList(10f, 10f);
		network.retroPropagateError(expectedValues);
	}

}

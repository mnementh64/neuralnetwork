package net.mnementh64.neural;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import org.junit.Assert;
import org.junit.Test;

import com.fasterxml.jackson.databind.ObjectMapper;

import net.mnementh64.neural.model.ActivationFunction;
import net.mnementh64.neural.model.WeightInitFunction;

public class NetworkSaveAndLoadTest
{

	@Test
	public void networkSaveTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightInitFunction.UNIT)
				.setLearningRate(0.5f)
				.addLayer(2)
				.addLayer(5, ActivationFunction.SIGMOID)
				.addLayer(1, ActivationFunction.SIGMOID)
				.build();
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		String output = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(network);
		System.out.println(output);
		String expectedOutput = TestUtils.loadResource("serialize1.json");
		Assert.assertTrue(output.equals(expectedOutput));
	}

	@Test
	public void networkLoadTest() throws Exception
	{
		String descriptor = TestUtils.loadResource("serialize1.json");
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		Network network = mapper.readValue(descriptor, Network.class);
		System.out.println(network);

		Assert.assertTrue(network.learningRate == 0.5f);
		Assert.assertTrue(network.size() == 3);
		Assert.assertTrue(network.getLayerSize(0) == 2);
		Assert.assertTrue(network.getLayerSize(1) == 5);
		Assert.assertTrue(network.getLayerSize(2) == 1);
	}

}

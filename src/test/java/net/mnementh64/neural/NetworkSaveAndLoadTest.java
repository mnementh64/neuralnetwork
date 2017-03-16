package net.mnementh64.neural;

import org.junit.Assert;
import org.junit.Test;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import net.mnementh64.neural.model.activation.ActivationUtils;
import net.mnementh64.neural.model.activation.SigmoideFunction;
import net.mnementh64.neural.model.activation.TanhFunction;
import net.mnementh64.neural.model.layer.Layer;
import net.mnementh64.neural.model.weight.RandomGaussianFunction;
import net.mnementh64.neural.model.weight.RandomUniformFunction;
import net.mnementh64.neural.model.weight.WeightUtils;

public class NetworkSaveAndLoadTest
{

	@Test
	public void networkSaveTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightUtils.unitFunction)
				.setLearningRate(0.5f)
				.addLayer(2)
				.addLayer(5, ActivationUtils.sigmoid)
				.addLayer(1, ActivationUtils.sigmoid)
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
		Assert.assertTrue(network.getLayerType(0).equals(Layer.Type.INPUT));
		Assert.assertTrue(network.getLayerSize(1) == 5);
		Assert.assertTrue(network.getLayerType(1).equals(Layer.Type.HIDDEN));
		Assert.assertTrue(network.getLayerSize(2) == 1);
		Assert.assertTrue(network.getLayerType(2).equals(Layer.Type.OUTPUT));
	}

	@Test
	public void networkLoadTestTypes() throws Exception
	{
		String descriptor = TestUtils.loadResource("serializeTypes.json");
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		Network network = mapper.readValue(descriptor, Network.class);
		System.out.println(network);

		Assert.assertTrue(network.getLayerActivationfunction(0) == null);
		Assert.assertTrue(network.getLayerActivationfunction(1) instanceof SigmoideFunction);
		Assert.assertTrue(network.getLayerActivationfunction(2) instanceof TanhFunction);
		Assert.assertTrue(network.getLayerWeightInitfunction(0) instanceof RandomUniformFunction);
		Assert.assertTrue(network.getLayerWeightInitfunction(1) instanceof RandomGaussianFunction);
		Assert.assertTrue(network.getLayerWeightInitfunction(2) == null);
	}

	@Test
	public void rnnSaveTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightUtils.unitFunction)
				.setLearningRate(0.5f)
				.addLayer(2)
				.addRecurrentLayer(5, ActivationUtils.sigmoid)
				.addLayer(1, ActivationUtils.sigmoid)
				.build();
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		String output = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(network);
		System.out.println(output);
		String expectedOutput = TestUtils.loadResource("rnn1.json");
		Assert.assertTrue(output.equals(expectedOutput));
	}

	@Test
	public void rnnLoadTest() throws Exception
	{
		String descriptor = TestUtils.loadResource("rnn1.json");
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		Network network = mapper.readValue(descriptor, Network.class);
		System.out.println(network);

		Assert.assertTrue(network.learningRate == 0.5f);
		Assert.assertTrue(network.size() == 3);
		Assert.assertTrue(network.getLayerSize(0) == 2);
		Assert.assertTrue(network.getLayerType(0).equals(Layer.Type.INPUT));
		Assert.assertTrue(network.getLayerSize(1) == 5);
		Assert.assertTrue(network.getLayerType(1).equals(Layer.Type.RECURRENT));
		Assert.assertTrue(network.getLayerSize(2) == 1);
		Assert.assertTrue(network.getLayerType(2).equals(Layer.Type.OUTPUT));
	}
}

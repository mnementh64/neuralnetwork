package net.mnementh64.neural;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Test;

import net.mnementh64.neural.model.activation.ActivationUtils;
import net.mnementh64.neural.model.layer.Layer;
import net.mnementh64.neural.model.weight.UnitFunction;
import net.mnementh64.neural.model.weight.WeightUtils;

public class RNNTest
{

	@Test
	public void rnnBuildTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightUtils.unitFunction)
				.addLayer(2)
				.addRecurrentLayer(5, ActivationUtils.sigmoid)
				.addLayer(8, ActivationUtils.sigmoid)
				.addLayer(1, ActivationUtils.sigmoid)
				.build();

		Assert.assertTrue(network.size() == 4);

		Assert.assertTrue(network.getLayerSize(0) == 2);
		Assert.assertTrue(network.getLayerType(0).equals(Layer.Type.INPUT));
		Assert.assertTrue(network.getLayerSize(1) == 5);
		Assert.assertTrue(network.getLayerType(1).equals(Layer.Type.RECURRENT));
		Assert.assertTrue(network.getLayerSize(2) == 8);
		Assert.assertTrue(network.getLayerType(2).equals(Layer.Type.HIDDEN));
		Assert.assertTrue(network.getLayerSize(3) == 1);
		Assert.assertTrue(network.getLayerType(3).equals(Layer.Type.OUTPUT));
	}

	@Test
	public void rnnFeedForwardTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(new UnitFunction(0.5))
				.addLayer(2)
				.addRecurrentLayer(3, ActivationUtils.identity)
				.addLayer(1, ActivationUtils.identity)
				.build();

		List<Double> input = Arrays.asList(2.0, 3.0);
		List<Double> output = network.feedForward(input);
		Assert.assertNotNull(output);
		Assert.assertTrue(output.size() == 1);
		Assert.assertTrue(output.get(0) == 3.75);

		output = network.feedForward(Arrays.asList(1.0, 2.0));
		Assert.assertTrue(output.get(0) == 7.875);

		output = network.feedForward(Arrays.asList(0.0, -1.0));
		Assert.assertTrue(output.get(0) == 11.0625);
	}

	@Test
	public void rnnRetroPropagationTest() throws Exception
	{
		String descriptor = TestUtils.loadResource("simpleRnn1.json");
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		Network network = mapper.readValue(descriptor, Network.class);

		List<Double> input = Arrays.asList(2.0, -1.0);
		List<Double> output = network.feedForward(input);

		List<Double> expectedValues = Collections.singletonList(1.0);
		network.retroPropagateError(expectedValues);

		String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(network);
		System.out.println(json);
		String expectedJson = TestUtils.loadResource("simpleRnn1-backprop.json");
		Assert.assertTrue(json.equals(expectedJson));
	}

}

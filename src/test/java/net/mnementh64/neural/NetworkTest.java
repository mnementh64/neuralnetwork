package net.mnementh64.neural;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import net.mnementh64.neural.model.DataRow;
import net.mnementh64.neural.model.NetworkRunStats;
import net.mnementh64.neural.model.activation.ActivationUtils;
import net.mnementh64.neural.model.weight.WeightUtils;

public class NetworkTest
{

	@Test
	public void networkBuildTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightUtils.unitFunction)
				.addLayer(2)
				.addLayer(5, ActivationUtils.sigmoid)
				.addLayer(1, ActivationUtils.sigmoid)
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
				.setWeightInitFunction(WeightUtils.unitFunction)
				.addLayer(2)
				.addLayer(3, ActivationUtils.sigmoid)
				.addLayer(1, ActivationUtils.sigmoid)
				.build();

		List<Double> input = Collections.singletonList(3.0);
		network.feedForward(input);
	}

	@Test
	public void networkFeedForwardTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightUtils.unitFunction)
				.addLayer(2)
				.addLayer(3, ActivationUtils.identity)
				.addLayer(1, ActivationUtils.identity)
				.build();

		List<Double> input = Arrays.asList(2.0, 3.0);
		List<Double> output = network.feedForward(input);

		Assert.assertNotNull(output);
		Assert.assertTrue(output.size() == 1);
		Assert.assertTrue(output.get(0) == 15);
	}

	@Test
	public void networkComplexFeedForwardTest() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightUtils.unitFunction)
				.addLayer(10)
				.addLayer(32, ActivationUtils.sigmoid)
				.addLayer(24, ActivationUtils.sigmoid)
				.addLayer(124, ActivationUtils.sigmoid)
				.addLayer(234, ActivationUtils.sigmoid)
				.addLayer(23, ActivationUtils.sigmoid)
				.build();

		List<Double> input = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
		List<Double> output = network.feedForward(input);

		Assert.assertNotNull(output);
		Assert.assertTrue(output.size() == 23);
	}

	@Test(expected = Exception.class)
	public void networkRetroPropagateTestBadSize() throws Exception
	{
		Network network = new Network.Builder()
				.setWeightInitFunction(WeightUtils.unitFunction)
				.addLayer(2)
				.addLayer(3, ActivationUtils.sigmoid)
				.addLayer(1, ActivationUtils.sigmoid)
				.build();

		List<Double> input = Arrays.asList(2.0, 3.0);
		network.feedForward(input);
		List<Double> expectedValues = Arrays.asList(10.0, 10.0);
		network.retroPropagateError(expectedValues);
	}

	@Test
	public void sherbrookFeedForwardTest() throws Exception
	{
		String descriptor = TestUtils.loadResource("sherbrook1.json");
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		Network network = mapper.readValue(descriptor, Network.class);

		List<Double> input = Arrays.asList(2.0, -1.0);
		List<Double> output = network.feedForward(input);

		Assert.assertNotNull(output);
		Assert.assertEquals(0.648, output.get(0), 0.001);
	}

	@Test
	public void sherbrookRetroPropagationTest() throws Exception
	{
		String descriptor = TestUtils.loadResource("sherbrook1.json");
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		Network network = mapper.readValue(descriptor, Network.class);

		List<Double> input = Arrays.asList(2.0, -1.0);
		List<Double> output = network.feedForward(input);

		List<Double> expectedValues = Collections.singletonList(1.0);
		network.retroPropagateError(expectedValues);

		String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(network);
		System.out.println(json);
		String expectedJson = TestUtils.loadResource("sherbrook1-step1.json");
		Assert.assertTrue(json.equals(expectedJson));
	}

	@Test
	public void sherbrookRun1Test() throws Exception
	{
		String descriptor = TestUtils.loadResource("sherbrook1.json");
		ObjectMapper mapper = new ObjectMapper();
		mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
		Network network = mapper.readValue(descriptor, Network.class);

		DataRow dataRow1 = new DataRow(Arrays.asList(2.0, -1.0), Collections.singletonList(1.0));
		DataRow dataRow2 = new DataRow(Arrays.asList(3.0, -2.0), Collections.singletonList(1.5));
		NetworkRunStats stats = NetworkRunner.of(network).run(Arrays.asList(dataRow1, dataRow2), 0.5, 1000, 1);

		Assert.assertTrue(stats.nbIterations >= 300);
		Assert.assertTrue(stats.error == 0);
	}
}

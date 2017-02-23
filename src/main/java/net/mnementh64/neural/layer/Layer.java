package net.mnementh64.neural.layer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import net.mnementh64.neural.Node;
import net.mnementh64.neural.WeightUtils;

public abstract class Layer
{

	/**
	 *  ordered collection of nodes
	 */
	List<Node> nodes;
	/**
	 * Activation function of all nodes : sigmoide, hyperbolic tangent, ...
	 */
	public ActivationFunction activationFunction;
	/**
	 * weights between layer nodes and next layer nodes
	 */
	float[][] weightsToNext;

	protected Layer(ActivationFunction activationFunction, int nbNodes)
	{
		this.activationFunction = activationFunction;

		// create nodes
		nodes = new ArrayList<>(nbNodes);
		IntStream.rangeClosed(1, nbNodes)
				.forEach(i -> nodes.add(new Node()));
	}

	public void linkTo(Layer nextLayer, WeightInitFunction weightInitFunction) throws Exception
	{
		weightsToNext = WeightUtils.init(this.nodes.size(), nextLayer.nodes.size(), weightInitFunction);
	}

	public int size()
	{
		return nodes.size();
	}

	public void feedForward(Layer previousLayer) throws Exception
	{
		// compute each current layer's node's value
		for (int j = 0; j < nodes.size(); j++)
		{
			Node node = nodes.get(j);
			node.input = previousLayer.computeOutputToNode(j);
			node.value = applyActivationFunction(node.input, activationFunction);
		}
	}

	public List<Float> getOutput()
	{
		return nodes.stream()
				.map(n -> n.value)
				.collect(Collectors.toList());
	}

	private float applyActivationFunction(float input, ActivationFunction activationFunction) throws Exception
	{
		switch (activationFunction)
		{
			case IDENTITY:
				return input;

			case SIGMOID:
				// TODO
				return input;
		}
		throw new Exception("Unsupported activation function : " + activationFunction);
	}

	private float computeOutputToNode(int nextLayerNodeIndex) throws Exception
	{
		float value = 0;

		for (int i = 0; i < nodes.size(); i++)
			value += nodes.get(i).value * weightsToNext[i][nextLayerNodeIndex];

		return value;
	}

	public void init(List<Float> input) throws Exception
	{
		if (input.size() != size())
			throw new Exception("Input values are bad sized for this layer : get " + input.size() + " items and expected " + size());

		IntStream.range(0, input.size())
				.forEach(i -> nodes.get(i).value = input.get(i));
	}
}

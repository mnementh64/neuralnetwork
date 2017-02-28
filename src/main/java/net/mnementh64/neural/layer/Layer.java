package net.mnementh64.neural.layer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import net.mnementh64.neural.Node;
import net.mnementh64.neural.WeightUtils;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes(
{ @JsonSubTypes.Type(value = InputLayer.class), @JsonSubTypes.Type(value = HiddenLayer.class), @JsonSubTypes.Type(value = OutputLayer.class), })
@JsonInclude(JsonInclude.Include.NON_NULL)
public abstract class Layer
{

	/**
	 *  ordered collection of nodes
	 */
	@JsonIgnore
	List<Node> nodes;
	/**
	 * Activation function of all nodes : sigmoide, hyperbolic tangent, ...
	 */
	@JsonProperty
	ActivationFunction activationFunction;
	/**
	 * weights between layer nodes and next layer nodes
	 */
	@JsonProperty
	float[][] weightsToNext;

	Layer()
	{
	}

	Layer(ActivationFunction activationFunction, int nbNodes)
	{
		this.activationFunction = activationFunction;

		// create nodes
		setNbNodes(nbNodes);
	}

	public void linkTo(Layer nextLayer, WeightInitFunction weightInitFunction) throws Exception
	{
		weightsToNext = WeightUtils.init(this.nodes.size(), nextLayer.nodes.size(), weightInitFunction);
	}

	public void init(List<Float> input) throws Exception
	{
		if (input.size() != getNbNodes())
			throw new Exception("Input values are bad sized for this layer : get " + input.size() + " items and expected " + getNbNodes());

		IntStream.range(0, input.size())
				.forEach(i -> nodes.get(i).value = input.get(i));
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

	public void computeDelta(Layer nextLayer) throws Exception
	{
		// compute each current layer's node's delta
		for (int j = 0; j < nodes.size(); j++)
		{
			Node node = nodes.get(j);
			float delta = nextLayer.computeWeightedDelta(weightsToNext[j]);
			node.delta = delta * applyActivationDerivativeFunction(node.input, activationFunction);
		}
	}

	public void adjustWeights(Layer nextLayer, float learningRate) throws Exception
	{
		for (int i = 0; i < nodes.size(); i++)
		{
			Node node = nodes.get(i);

			for (int j = 0; j < nextLayer.getNbNodes(); j++)
			{
				float nextDelta = nextLayer.getDelta(j);
				weightsToNext[i][j] += learningRate * node.value * nextDelta;
			}
		}
	}

	public List<Float> output()
	{
		return nodes.stream()
				.map(n -> n.value)
				.collect(Collectors.toList());
	}

	public int getNbNodes()
	{
		return nodes.size();
	}

	public void setNbNodes(int nbNodes)
	{
		nodes = new ArrayList<>(nbNodes);
		IntStream.rangeClosed(1, nbNodes)
				.forEach(i -> nodes.add(new Node()));
	}

	private float getDelta(int j) throws Exception
	{
		return nodes.get(j).delta;
	}

	private float computeWeightedDelta(float[] weights) throws Exception
	{
		float value = 0;
		for (int i = 0; i < nodes.size(); i++)
			value += weights[i] * nodes.get(i).delta;

		return value;
	}

	private float applyActivationFunction(float input, ActivationFunction activationFunction) throws Exception
	{
		switch (activationFunction)
		{
			case IDENTITY:
				return input;

			case SIGMOID:
				return sigmoide(input);
		}
		throw new Exception("Unsupported activation function : " + activationFunction);
	}

	private float applyActivationDerivativeFunction(float input, ActivationFunction activationFunction) throws Exception
	{
		switch (activationFunction)
		{
			case IDENTITY:
				return 1;

			case SIGMOID:
				return sigmoide(input) * (1.0f - sigmoide(input));
		}
		throw new Exception("Unsupported derivative activation function : " + activationFunction);
	}

	private float sigmoide(float x)
	{
		return (float) (1.0 / (1.0 + Math.exp(-x)));
	}

	private float computeOutputToNode(int nextLayerNodeIndex) throws Exception
	{
		float value = 0;

		for (int i = 0; i < nodes.size(); i++)
			value += nodes.get(i).value * weightsToNext[i][nextLayerNodeIndex];

		return value;
	}
}

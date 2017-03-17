package net.mnementh64.neural.model.layer;

import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import net.mnementh64.neural.model.Node;
import net.mnementh64.neural.model.activation.ActivationFunction;
import net.mnementh64.neural.model.activation.ActivationUtils;
import net.mnementh64.neural.model.weight.WeightInitFunction;

public class RecurrentLayer extends Layer
{

	/**
	 * weights between layer nodes and itself a next iteration
	 */
	@JsonProperty
	double[][] weightsToItself;
	@JsonIgnore
	double[] previousValues;
	@JsonIgnore
	public double[] previousDeltas;

	public RecurrentLayer()
	{
		super(Type.RECURRENT);
	}

	public RecurrentLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(Type.RECURRENT, activationFunction == null ? ActivationUtils.sigmoid : activationFunction, nbNodes);
	}

	public void linkTo(Layer nextLayer, WeightInitFunction weightInitFunction) throws Exception
	{
		super.linkTo(nextLayer, weightInitFunction);

		// also init weight to itself
		this.weightsToItself = weightInitFunction.init(this.nodes.size(), this.nodes.size());
	}

	public void feedForward(Layer previousLayer) throws Exception
	{
		if (Objects.isNull(previousValues))
			previousValues = new double[nodes.size()];

		// compute each current layer's node's value
		for (int j = 0; j < nodes.size(); j++)
		{
			Node node = nodes.get(j);
			// recurrent layer node input is computed from :
			// - previous layer nodes in network
			node.input = previousLayer.computeOutputToNode(j);
			// - previous execution of current layer (recursion)
			node.input += computeRecurrentForward(j);
			node.value = activationFunction.apply(node.input);
		}

		// store all these nodes values as input for next execution
		for (int i = 0; i < nodes.size(); i++)
			previousValues[i] = nodes.get(i).value;
	}

	public void computeDelta(Layer nextLayer) throws Exception
	{
		if (Objects.isNull(previousDeltas))
			previousDeltas = new double[nodes.size()];

		// compute each current layer's node's delta
		for (int j = 0; j < nodes.size(); j++)
		{
			Node node = nodes.get(j);
			// recurrent layer delta is computed from :
			// - next layer deltas
			node.delta = nextLayer.computeWeightedDelta(weightsToNext[j]);
			// - previous execution of next layer (recursion)
			node.delta += computeRecurrentDelta(j);
			node.delta *= activationFunction.applyDerivative(node.input);
		}

		// store all these nodes deltas as deltas for next execution
		for (int i = 0; i < nodes.size(); i++)
			previousDeltas[i] = nodes.get(i).delta;
	}

	public void adjustWeights(Layer nextLayer, double learningRate) throws Exception
	{
		super.adjustWeights(nextLayer, learningRate);
		for (int i = 0; i < nodes.size(); i++)
		{
			Node node = nodes.get(i);
			for (int j = 0; j < nodes.size(); j++)
			{
				weightsToItself[i][j] += learningRate * node.value * previousDeltas[j];
			}
		}
	}

	private double computeRecurrentDelta(int nodeIndex)
	{
		double value = 0;
		for (int i = 0; i < previousDeltas.length; i++)
			value += previousDeltas[i] * weightsToItself[i][nodeIndex];

		return value;
	}

	private double computeRecurrentForward(int nodeIndex)
	{
		double value = 0;

		for (int i = 0; i < previousValues.length; i++)
			value += previousValues[i] * weightsToItself[i][nodeIndex];

		return value;
	}

}

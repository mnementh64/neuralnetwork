package net.mnementh64.neural.model.layer;

import com.fasterxml.jackson.annotation.JsonIgnore;

import net.mnementh64.neural.model.Node;
import net.mnementh64.neural.model.activation.ActivationFunction;
import net.mnementh64.neural.model.activation.ActivationUtils;
import net.mnementh64.neural.model.weight.WeightInitFunction;

public class RecurrentLayer extends Layer
{

	/**
	 * weights between layer nodes and itself a next iteration
	 */
	@JsonIgnore
	double[][] weightsToItself;
	@JsonIgnore
	double[] previousValues;

	public RecurrentLayer()
	{
		super(Type.RECURRENT);
	}

	public RecurrentLayer(ActivationFunction activationFunction, int nbNodes)
	{
		super(Type.RECURRENT, activationFunction == null ? ActivationUtils.sigmoid : activationFunction, nbNodes);
		previousValues = new double[nodes.size()];
	}

	public void linkTo(Layer nextLayer, WeightInitFunction weightInitFunction) throws Exception
	{
		super.linkTo(nextLayer, weightInitFunction);

		// also init weight to itself
		this.weightsToItself = weightInitFunction.init(this.nodes.size(), this.nodes.size());
	}

	public void feedForward(Layer previousLayer) throws Exception
	{
		// compute each current layer's node's value
		for (int j = 0; j < nodes.size(); j++)
		{
			Node node = nodes.get(j);
			// recurrent layer node input is computed from :
			// - previous layer nodes in netwark
			node.input = previousLayer.computeOutputToNode(j);
			// - previous execution of current layer (recursivity)
			node.input += computeRecurrentForward(j);
			node.value = activationFunction.apply(node.input);
		}

		// store all these nodes' values as input for next execution
		for (int i = 0; i < nodes.size(); i++)
			previousValues[i] = nodes.get(i).value;
	}

	private double computeRecurrentForward(int nodeIndex)
	{
		double value = 0;

		for (int i = 0; i < previousValues.length; i++)
			value += previousValues[i] * weightsToItself[i][nodeIndex];

		return value;
	}

}

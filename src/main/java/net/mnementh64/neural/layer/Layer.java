package net.mnementh64.neural.layer;

import java.util.ArrayList;
import java.util.List;
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
	ActivationFunction activationFunction;
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
}

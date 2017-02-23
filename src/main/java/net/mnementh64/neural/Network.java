package net.mnementh64.neural;

import java.util.ArrayList;
import java.util.List;

import net.mnementh64.neural.layer.ActivationFunction;
import net.mnementh64.neural.layer.HiddenLayer;
import net.mnementh64.neural.layer.InputLayer;
import net.mnementh64.neural.layer.Layer;
import net.mnementh64.neural.layer.OutputLayer;
import net.mnementh64.neural.layer.WeightInitFunction;

class Network
{

	private List<Layer> layers = new ArrayList<>();

	private Network()
	{
	}

	List<Float> feedForward(List<Float> input) throws Exception
	{
		// check input size
		if (input.size() != getInputSize())
			throw new Exception("Input vector is expected to be of size " + getInputSize());

		// feed first layer with input values
		layers.get(0).init(input);

		// Feed forward each layer after first
		Layer previousLayer = null;
		for (Layer layer : layers)
		{
			layer.feedForward(previousLayer);
			previousLayer = layer;
		}

		return layers.get(layers.size() - 1).getOutput();
	}

	void retroPropagateError(List<Float> expectedValues) throws Exception
	{
		// check the expected values size
		if (expectedValues.size() != getOutputSize())
			throw new Exception("Expected values vector must be of size " + getOutputSize());

		// start from the last layer
	}

	private void setLayers(List<Layer> layers)
	{
		this.layers = layers;
	}

	int size()
	{
		return layers.size();
	}

	/**
	 * Get a layer size
	 * @param layerIndex : 0 based index
	 * @return
	 */
	int getLayerSize(int layerIndex)
	{
		return layers.get(layerIndex).size();
	}

	int getInputSize()
	{
		return getLayerSize(0);
	}

	int getOutputSize()
	{
		return getLayerSize(size() - 1);
	}

	static class Builder
	{

		private List<LayerDescriptor> layerDescriptors = new ArrayList<>();
		private WeightInitFunction weightInitFunction;

		private class LayerDescriptor
		{

			int nbNodes;
			ActivationFunction activationFunction;
		}

		Builder setWeightInitFunction(WeightInitFunction weightInitFunction)
		{
			this.weightInitFunction = weightInitFunction;
			return this;
		}

		Builder addLayer(int nbNodes)
		{
			LayerDescriptor layerDescriptor = new LayerDescriptor();
			layerDescriptor.nbNodes = nbNodes;
			layerDescriptors.add(layerDescriptor);
			return this;
		}

		Builder addLayer(int nbNodes, ActivationFunction activationFunction)
		{
			LayerDescriptor layerDescriptor = new LayerDescriptor();
			layerDescriptor.nbNodes = nbNodes;
			layerDescriptor.activationFunction = activationFunction;
			layerDescriptors.add(layerDescriptor);
			return this;
		}

		Network build() throws Exception
		{
			checkRequirements();

			Network network = new Network();

			// -------------------------------------------------------------------
			// Create layers
			List<Layer> layers = new ArrayList<>();
			// first layer
			LayerDescriptor layerDescriptor = layerDescriptors.get(0);
			Layer firstLayer = new InputLayer(layerDescriptor.nbNodes); // no need of any activation function
			layers.add(firstLayer);

			// hidden layers
			for (int i = 1; i < (layerDescriptors.size() - 1); i++)
			{
				layerDescriptor = layerDescriptors.get(i);
				Layer hiddenLayer = new HiddenLayer(layerDescriptor.activationFunction, layerDescriptor.nbNodes);
				layers.add(hiddenLayer);
			}

			// output layer
			layerDescriptor = layerDescriptors.get(layerDescriptors.size() - 1);
			Layer outputLayer = new OutputLayer(layerDescriptor.activationFunction, layerDescriptor.nbNodes);
			layers.add(outputLayer);

			// --------------------------------------------------------------------
			// link layers
			Layer previousLayer = null;
			for (Layer layer : layers)
			{
				if (previousLayer != null)
					previousLayer.linkTo(layer, this.weightInitFunction);
				previousLayer = layer;
			}

			network.setLayers(layers);
			return network;
		}

		private void checkRequirements() throws Exception
		{
			// at least 2 layers
			if (layerDescriptors.size() <= 2)
				throw new Exception("Can't create a Neural network with less than 2 layers");
		}
	}

}

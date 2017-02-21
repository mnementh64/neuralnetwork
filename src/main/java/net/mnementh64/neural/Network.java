package net.mnementh64.neural;

import java.util.ArrayList;
import java.util.List;

import net.mnementh64.neural.layer.ActivationFunction;
import net.mnementh64.neural.layer.HiddenLayer;
import net.mnementh64.neural.layer.InputLayer;
import net.mnementh64.neural.layer.Layer;
import net.mnementh64.neural.layer.OutputLayer;
import net.mnementh64.neural.layer.WeightInitFunction;

public class Network
{

	List<Layer> layers = new ArrayList<>();

	private Network()
	{
	}

	private void setLayers(List<Layer> layers)
	{
		this.layers = layers;
	}

	static class Builder
	{

		List<LayerDescriptor> layerDescriptors = new ArrayList<>();
		WeightInitFunction weightInitFunction;

		private class LayerDescriptor
		{

			int nbNodes;
			ActivationFunction activationFunction;
		}

		public Builder setWeightInitFunction(WeightInitFunction weightInitFunction)
		{
			this.weightInitFunction = weightInitFunction;
			return this;
		}

		public Builder addLayer(int nbNodes, ActivationFunction activationFunction)
		{
			LayerDescriptor layerDescriptor = new LayerDescriptor();
			layerDescriptor.nbNodes = nbNodes;
			layerDescriptor.activationFunction = activationFunction;
			layerDescriptors.add(layerDescriptor);
			return this;
		}

		public Network build() throws Exception
		{
			checkRequirements();

			Network network = new Network();

			// -------------------------------------------------------------------
			// Create layers
			List<Layer> layers = new ArrayList<>();
			// first layer
			LayerDescriptor layerDescriptor = layerDescriptors.get(0);
			Layer firstLayer = new InputLayer(layerDescriptor.activationFunction, layerDescriptor.nbNodes);
			layers.add(firstLayer);

			// hidden layers
			for (int i = 1; i < layerDescriptors.size(); i++)
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

package net.mnementh64.neural;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonProperty;

import net.mnementh64.neural.model.activation.ActivationFunction;
import net.mnementh64.neural.model.layer.HiddenLayer;
import net.mnementh64.neural.model.layer.InputLayer;
import net.mnementh64.neural.model.layer.Layer;
import net.mnementh64.neural.model.layer.OutputLayer;
import net.mnementh64.neural.model.layer.RecurrentLayer;
import net.mnementh64.neural.model.weight.WeightInitFunction;

public class Network
{

	public final static Logger L = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	@JsonProperty
	List<Layer> layers = new ArrayList<>();
	@JsonProperty
	double learningRate = 0.01f;

	public Network()
	{
	}

	public List<Double> feedForward(List<Double> input) throws Exception
	{
		// feed first layer with input values
		layers.get(0).init(input);

		// Feed forward each layer after first
		Layer previousLayer = null;
		for (Layer layer : layers)
		{
			layer.feedForward(previousLayer);
			previousLayer = layer;
		}

		return layers.get(layers.size() - 1).output();
	}

	public void retroPropagateError(List<Double> expectedValues) throws Exception
	{
		// compute error for the last layer
		((OutputLayer) layers.get(layers.size() - 1)).computeError(expectedValues);

		// From the pre-last layer : compute layers's nodes' delta
		Layer nextLayer = layers.get(layers.size() - 1);
		for (int i = layers.size() - 2; i >= 1; i--)
		{
			layers.get(i).computeDelta(nextLayer);
			nextLayer = layers.get(i);
		}

		// From the pre-last layer : adjust layers's weights
		nextLayer = layers.get(layers.size() - 1);
		for (int i = layers.size() - 2; i >= 0; i--)
		{
			layers.get(i).adjustWeights(nextLayer, learningRate);
			nextLayer = layers.get(i);
		}
	}

	public int size()
	{
		return layers.size();
	}

	/**
	 * Get a layer getNbNodes
	 * @param layerIndex : 0 based index
	 * @return
	 */
	public int getLayerSize(int layerIndex)
	{
		return layers.get(layerIndex).getNbNodes();
	}

	/**
	 * Get a layer weight init function
	 * @param layerIndex : 0 based index
	 * @return
	 */
	public WeightInitFunction getLayerWeightInitfunction(int layerIndex)
	{
		return layers.get(layerIndex).weightInitFunction;
	}

	/**
	 * Get a layer activation function
	 * @param layerIndex : 0 based index
	 * @return
	 */
	public ActivationFunction getLayerActivationfunction(int layerIndex)
	{
		return layers.get(layerIndex).activationFunction;
	}

	/**
	 * Get a layer weight init function
	 * @param layerIndex : 0 based index
	 * @return
	 */
	public Layer.Type getLayerType(int layerIndex)
	{
		return layers.get(layerIndex).type;
	}

	private void setLayers(List<Layer> layers)
	{
		this.layers = layers;
	}

	public static class Builder
	{

		private List<LayerDescriptor> layerDescriptors = new ArrayList<>();
		private WeightInitFunction weightInitFunction;
		private double learningRate = 0.01f;

		private static class LayerDescriptor
		{

			boolean isRecurrent = false;
			int nbNodes;
			ActivationFunction activationFunction;

			static LayerDescriptor from(int nbNodes, ActivationFunction activationFunction)
			{
				LayerDescriptor layerDescriptor = new LayerDescriptor();
				layerDescriptor.nbNodes = nbNodes;
				layerDescriptor.activationFunction = activationFunction;
				return layerDescriptor;
			}
		}

		public Builder setWeightInitFunction(WeightInitFunction weightInitFunction)
		{
			this.weightInitFunction = weightInitFunction;
			return this;
		}

		public Builder setLearningRate(float learningRate)
		{
			this.learningRate = learningRate;
			return this;
		}

		public Builder addLayer(int nbNodes)
		{
			LayerDescriptor layerDescriptor = new LayerDescriptor();
			layerDescriptor.nbNodes = nbNodes;
			layerDescriptors.add(layerDescriptor);
			return this;
		}

		public Builder addRecurrentLayer(int nbNodes, ActivationFunction activationFunction)
		{
			LayerDescriptor layerDescriptor = LayerDescriptor.from(nbNodes, activationFunction);
			layerDescriptor.isRecurrent = true;
			layerDescriptors.add(layerDescriptor);
			return this;
		}

		public Builder addLayer(int nbNodes, ActivationFunction activationFunction)
		{
			layerDescriptors.add(LayerDescriptor.from(nbNodes, activationFunction));
			return this;
		}

		public Network build() throws Exception
		{
			checkRequirements();

			Network network = new Network();
			network.learningRate = learningRate;

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
				Layer hiddenLayer = layerDescriptor.isRecurrent ? new RecurrentLayer(layerDescriptor.activationFunction, layerDescriptor.nbNodes)
						: new HiddenLayer(layerDescriptor.activationFunction, layerDescriptor.nbNodes);
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

package net.mnementh64.neural;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonProperty;

import net.mnementh64.neural.model.ActivationFunction;
import net.mnementh64.neural.model.DataRow;
import net.mnementh64.neural.model.NetworkRunStats;
import net.mnementh64.neural.model.WeightInitFunction;
import net.mnementh64.neural.model.layer.HiddenLayer;
import net.mnementh64.neural.model.layer.InputLayer;
import net.mnementh64.neural.model.layer.Layer;
import net.mnementh64.neural.model.layer.OutputLayer;
import net.mnementh64.neural.utils.DataRowUtils;

class Network
{

	public final static Logger L = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	@JsonProperty
	List<Layer> layers = new ArrayList<>();
	@JsonProperty
	float learningRate = 0.01f;

	public Network()
	{
	}

	NetworkRunStats run(List<DataRow> allData, float percentTraining, int maxIterations, int maxOvertraining) throws Exception
	{
		int nbIterations = 0;
		float totalError = Float.POSITIVE_INFINITY;
		float oldError = Float.POSITIVE_INFINITY;
		float totalGeneralizeError = Float.POSITIVE_INFINITY;
		float oldGeneralizeError = Float.POSITIVE_INFINITY;
		int nbOverTraining = 0;
		NetworkRunStats networkRunStats = new NetworkRunStats();

		// separate training / generalize sets
		List<DataRow> trainingSet = DataRowUtils.extractTrainingDataRow(allData, percentTraining);
		List<DataRow> generalizeSet = DataRowUtils.extractGeneralizeDataRow(allData, percentTraining);
		if (trainingSet.isEmpty())
			throw new Exception("Can't run a network without any training set !");
		networkRunStats.trainingSetSize = trainingSet.size();
		networkRunStats.generalizeSetSize = generalizeSet.size();

		for (int i = 1; i <= maxIterations; i++)
		{
			oldError = totalError;
			totalError = 0;
			oldGeneralizeError = totalGeneralizeError;
			totalGeneralizeError = 0;

			// training
			for (DataRow data : trainingSet)
			{
				List<Float> output = this.feedForward(data.input);
				this.retroPropagateError(data.expectedOutput);

				for (int nb = 0; nb < output.size(); nb++)
				{
					double erreur = data.expectedOutput.get(nb) - output.get(nb);
					totalError += (erreur * erreur);
				}
			}

			// generalize
			for (DataRow data : generalizeSet)
			{
				List<Float> output = this.feedForward(data.input);
				this.retroPropagateError(data.expectedOutput);

				for (int nb = 0; nb < output.size(); nb++)
				{
					double erreur = data.expectedOutput.get(nb) - output.get(nb);
					totalGeneralizeError += (erreur * erreur);
				}
			}

			// detect overtraining
			nbOverTraining += totalGeneralizeError > oldGeneralizeError ? 1 : 0;

			// Learning rate adjustement ?
			learningRate /= totalError > oldError ? 2.0 : 1.0;

			// check stop conditions
			nbIterations++;
			if (nbIterations >= maxIterations)
			{
				L.info("Max iterations ! Iteration n°" + nbIterations + " - total error : " + totalError + " - Generalisation : " + totalGeneralizeError
						+ " - Learning rate : " + learningRate + " - Avg : " + Math.sqrt(totalError / trainingSet.size()));
				break;
			}
			if (totalError <= 0)
			{
				L.info("Error 0 reached ! Iteration n°" + nbIterations + " - total error : " + totalError + " - Generalisation : " + totalGeneralizeError
						+ " - Learning rate : " + learningRate + " - Avg : " + Math.sqrt(totalError / trainingSet.size()));
				break;
			}
			if (nbOverTraining >= maxOvertraining)
			{
				L.info("Overtraining ! Iteration n°" + nbIterations + " - total error : " + totalError + " - Generalisation : " + totalGeneralizeError
						+ " - Learning rate : " + learningRate + " - Avg : " + Math.sqrt(totalError / trainingSet.size()));
				break;
			}
		}

		networkRunStats.avgError = (float) Math.sqrt(totalError / trainingSet.size());
		networkRunStats.error = (float) totalError;
		networkRunStats.nbIterations = nbIterations;
		networkRunStats.overTrainingOccurences = nbOverTraining;
		networkRunStats.learningRate = learningRate;

//		System.out.println("Expected --> calculated");
//		{
//			PointND pointTest = new PointND(
//					"0.3333333333333333\t0.4791666666666667\t0.5\t0.3333333333333333\t0.4666666666666667\t0.0\t0.6\t0.5\t0.9280434782608695",
//					1);
//			double[] sorties = reseau.Evaluer(pointTest);
//			for (int i = 0; i < 1; i++)
//			{
//				System.out.println(pointTest.sorties[i] * 300000 + " --> " + sorties[i] * 300000);
//			}
//			System.out.println("Efficacité du réseau : " + (nbIterations / erreurTotale));
//			if (erreurTotale < erreurMin)
//			{
//				erreurMin = erreurTotale;
//				double distance = pointTest.entrees[1] * 6000;
//				double tempsExpected = pointTest.sorties[0] * distance * 80;
//				double tempsPredicted = sorties[0] * distance * 80;
//				writeFile("Iteration " + nbIterations + ", get error " + erreurMin + " and prediction is " + tempsPredicted + " compared to "
//						+ tempsExpected + " (Efficiency : " + (nbIterations / erreurTotale) + ")\n",
//						"/home/scaillet/workspace/IdeaProjects/perso/sniffhrstats/src/main/resources/reseauxNeurones/cheval/VIENNOISE.results.txt");
//			}
//		}
//
////		if (nbSurapprentissage >= MAX_SUR_APPRENTISSAGE)
////		{
////			System.out.println("Surapprentissage  --> restart ...");
////			init();
////			Lancer();
////		}
//
//		//Create plot with out data
//		plotDataset(donnees, reseau);

		return networkRunStats;
	}

	List<Float> feedForward(List<Float> input) throws Exception
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

	void retroPropagateError(List<Float> expectedValues) throws Exception
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

	private void setLayers(List<Layer> layers)
	{
		this.layers = layers;
	}

	int size()
	{
		return layers.size();
	}

	/**
	 * Get a layer getNbNodes
	 * @param layerIndex : 0 based index
	 * @return
	 */
	int getLayerSize(int layerIndex)
	{
		return layers.get(layerIndex).getNbNodes();
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
		private float learningRate = 0.01f;

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

		Builder setLearningRate(float learningRate)
		{
			this.learningRate = learningRate;
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

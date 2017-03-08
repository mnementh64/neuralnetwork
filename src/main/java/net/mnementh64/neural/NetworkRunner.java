package net.mnementh64.neural;

import java.lang.invoke.MethodHandles;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.mnementh64.neural.model.DataRow;
import net.mnementh64.neural.model.NetworkRunStats;
import net.mnementh64.neural.utils.DataRowUtils;

public class NetworkRunner
{

	private final static Logger L = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private Network network;

	public static NetworkRunner of(Network network)
	{
		NetworkRunner runner = new NetworkRunner();
		runner.network = network;



		return runner;
	}

	public NetworkRunStats run(List<DataRow> allData, float percentTraining, int maxIterations, int maxOvertraining) throws Exception
	{
		int nbIterations = 0;
		float totalError = Float.POSITIVE_INFINITY;
		float oldError = Float.POSITIVE_INFINITY;
		float totalGeneralizeError = Float.POSITIVE_INFINITY;
		float oldGeneralizeError = Float.POSITIVE_INFINITY;
		int nbOverTraining = 0;
		float learningRate = network.learningRate;
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
				List<Float> output = network.feedForward(data.input);
				network.retroPropagateError(data.expectedOutput);

				for (int nb = 0; nb < output.size(); nb++)
				{
					double erreur = data.expectedOutput.get(nb) - output.get(nb);
					totalError += (erreur * erreur);
				}
			}

			// generalize
			for (DataRow data : generalizeSet)
			{
				List<Float> output = network.feedForward(data.input);
				network.retroPropagateError(data.expectedOutput);

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
}

package net.mnementh64.neural;

import net.mnementh64.neural.model.DataRow;
import net.mnementh64.neural.model.NetworkRunStats;
import net.mnementh64.neural.utils.DataRowUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandles;
import java.util.List;

public class NetworkRunner {

    private final static Logger L = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    private Network network;

    public static NetworkRunner of(Network network) {
        NetworkRunner runner = new NetworkRunner();
        runner.network = network;

        return runner;
    }

    public NetworkRunStats run(List<DataRow> allData, double percentTraining, int maxIterations, int maxOvertraining) throws Exception {
        int nbIterations = 0;
        double totalError = Double.POSITIVE_INFINITY;
        double oldError = Double.POSITIVE_INFINITY;
        double totalGeneralizeError = Double.POSITIVE_INFINITY;
        double oldGeneralizeError = Double.POSITIVE_INFINITY;
        int nbOverTraining = 0;
        double learningRate = network.learningRate;
        NetworkRunStats networkRunStats = new NetworkRunStats();

        // separate training / generalize sets
        List<DataRow> trainingSet = DataRowUtils.extractTrainingDataRow(allData, percentTraining);
        List<DataRow> generalizeSet = DataRowUtils.extractGeneralizeDataRow(allData, percentTraining);
        if (trainingSet.isEmpty())
            throw new Exception("Can't run a network without any training set !");
        networkRunStats.trainingSetSize = trainingSet.size();
        networkRunStats.generalizeSetSize = generalizeSet.size();

        for (int i = 1; i <= maxIterations; i++) {
            oldError = totalError;
            totalError = 0;
            oldGeneralizeError = totalGeneralizeError;
            totalGeneralizeError = 0;

            // training
            for (DataRow data : trainingSet) {
                List<Double> output = network.feedForward(data.input);
                network.retroPropagateError(data.expectedOutput);

                for (int nb = 0; nb < output.size(); nb++) {
                    double erreur = data.expectedOutput.get(nb) - output.get(nb);
                    totalError += (erreur * erreur);
                }
            }

            // generalize
            for (DataRow data : generalizeSet) {
                List<Double> output = network.feedForward(data.input);
                network.retroPropagateError(data.expectedOutput);

                for (int nb = 0; nb < output.size(); nb++) {
                    double erreur = data.expectedOutput.get(nb) - output.get(nb);
                    totalGeneralizeError += (erreur * erreur);
                }
            }

            // detect overtraining
            nbOverTraining += totalGeneralizeError > oldGeneralizeError ? 1 : 0;

            // Learning rate adjustement ?
            learningRate /= totalError > oldError ? 1.2 : 1.0;

            L.info("Iteration n°" + nbIterations + " - total error : " + totalError + " - Generalisation : " + totalGeneralizeError
                    + " - Learning rate : " + learningRate);

            // check stop conditions
            nbIterations++;
            if (nbIterations >= maxIterations) {
                L.info("Max iterations ! Iteration n°" + nbIterations + " - total error : " + totalError + " - Generalisation : " + totalGeneralizeError
                        + " - Learning rate : " + learningRate + " - Avg : " + Math.sqrt(totalError / trainingSet.size()));
                break;
            }
            if (totalError <= 0) {
                L.info("Error 0 reached ! Iteration n°" + nbIterations + " - total error : " + totalError + " - Generalisation : " + totalGeneralizeError
                        + " - Learning rate : " + learningRate + " - Avg : " + Math.sqrt(totalError / trainingSet.size()));
                break;
            }
            if (nbOverTraining >= maxOvertraining) {
                L.info("Overtraining ! Iteration n°" + nbIterations + " - total error : " + totalError + " - Generalisation : " + totalGeneralizeError
                        + " - Learning rate : " + learningRate + " - Avg : " + Math.sqrt(totalError / trainingSet.size()));
                break;
            }
        }

        // update the network's learning rate
        network.learningRate = learningRate;

        networkRunStats.avgError = Math.sqrt(totalError / trainingSet.size());
        networkRunStats.avgGeneralizationError = Math.sqrt(totalGeneralizeError / generalizeSet.size());
        networkRunStats.error = totalError;
        networkRunStats.nbIterations = nbIterations;
        networkRunStats.overTrainingOccurences = nbOverTraining;
        networkRunStats.learningRate = learningRate;

        return networkRunStats;
    }

    public List<Double> predict(DataRow data) throws Exception {
        return network.feedForward(data.input);
    }
}

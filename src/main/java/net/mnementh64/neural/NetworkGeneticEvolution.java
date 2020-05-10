package net.mnementh64.neural;

import net.mnementh64.neural.model.layer.Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NetworkGeneticEvolution {

    private Random random = new Random(System.currentTimeMillis());

    public void checkSimilarity(Network network, Network other) {
        // both networks must be similar - same type, same number of layers and all layers with the same size / same activation function
        if ((other.layers.size() != network.layers.size())
        ) {
            throw new IllegalArgumentException("Can't crosssover 2 networks not similar");
        }

        // skip INPUT layer - genetic evolution only affect HIDDEN layers, so also rely on OUTPUT layer (for weights of the last layer)
        for (int i = 1; i < network.layers.size(); i++) {
            if (!network.layers.get(i).type.equals(other.layers.get(i).type) ||
                    network.layers.get(i).getNbNodes() != other.layers.get(i).getNbNodes() ||
                    !network.layers.get(i).activationFunction.equals(other.layers.get(i).activationFunction)) {
                throw new IllegalArgumentException("Can't crosssover 2 networks not similar");
            }
        }
    }

    /**
     * Single point binary crossover.
     * <p>
     * Create a new network with the same input / output and
     * 1) Random choice of a hidden layer.
     * 2) get network2's layers until this layer
     * 3) get network1's layers after this layer
     */
    public Network SPBX(Network network1, Network network2) {
        Network nextGenerationNetwork = new Network();
        List<Layer> newLayers = new ArrayList<>();

        // random selection of hidden layer to split
        int whichLayer = getRandomLayerForCrossover(network1.layers.size()); // can't be the input / output layers

        // copy all layers before the split one (including it) from the network 2
        for (int i = 0; i <= whichLayer; i++) {
            newLayers.add(network2.layers.get(i).clone());
        }

        // copy all layers after the split one from network 1
        for (int i = whichLayer + 1; i < network1.layers.size(); i++) {
            newLayers.add(network1.layers.get(i).clone());
        }

        nextGenerationNetwork.layers = newLayers;
        return nextGenerationNetwork;
    }

    public void mutate(Network network, double mutationRate, boolean applyExtremValue, double minValue, double maxValue) {
        if (mutationRate >= 1.0) {
            throw new IllegalArgumentException("Mutation rate should be strictly less than 1");
        }
        for (int layerIndex = 0; layerIndex < network.layers.size() - 1; layerIndex++) {
            Layer layer = network.layers.get(layerIndex);
            for (int i = 0; i < layer.getNbNodes(); i++) {
                for (int j = 0; j < network.layers.get(layerIndex + 1).getNbNodes(); j++) {
                    if (random.nextDouble() < mutationRate) {
                        layer.weightsToNext[i][j] += random.nextGaussian() / 5;

                        if (applyExtremValue) {
                            if (layer.weightsToNext[i][j] > maxValue) {
                                layer.weightsToNext[i][j] = maxValue;
                            }
                            if (layer.weightsToNext[i][j] < minValue) {
                                layer.weightsToNext[i][j] = minValue;
                            }
                        }
                    }
                }
            }
        }
    }

    int getRandomLayerForCrossover(int nbLayers) {
        // random selection of hidden layer to split
        return random.nextInt(nbLayers - 2) + 1;
    }
}

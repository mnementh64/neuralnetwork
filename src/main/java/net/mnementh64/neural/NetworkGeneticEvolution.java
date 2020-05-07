package net.mnementh64.neural;

import net.mnementh64.neural.model.layer.HiddenLayer;
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

        for (int i = 0; i < network.layers.size(); i++) {
            if (!network.layers.get(i).type.equals(other.layers.get(i).type) ||
                    network.layers.get(i).getNbNodes() != other.layers.get(i).getNbNodes() ||
                    !network.layers.get(i).activationFunction.equals(other.layers.get(i).activationFunction)) {
                throw new IllegalArgumentException("Can't crosssover 2 networks not similar");
            }
        }
    }

    public Network crossover(Network network, Network other) {
        Network nextGenerationNetwork = new Network();
        List<Layer> newLayers = new ArrayList<>();

        // crossover the network then mutate the weights

        // random selection of hidden layer to split
        int whichLayer = getRandomLayerForCrossover(network.layers.size()); // can't be the input / output layers

        // random selection of nodes to split from in this layer
        int whichNode = getRandomNodeForCrossover(whichLayer, network.layers.get(whichLayer).getNbNodes());

        // copy all layers before the split one from current network
        for (int i = 0; i < whichLayer; i++) {
            newLayers.add(network.layers.get(i).clone());
        }

        // TODO : in between, need to create a layer with some nodes from current network (before whichNode index) and last nodes from other network
        Layer originalLayer = network.layers.get(whichLayer);
        Layer mixLayer = new HiddenLayer(originalLayer.activationFunction, originalLayer.getNbNodes());
        mixLayer.weightsToNext = new double[originalLayer.weightsToNext.length][originalLayer.weightsToNext[0].length];
        // copy first part of the weights for layer from this network
        for (int i = 0; i < whichNode; i++) {
            mixLayer.weightsToNext[i] = originalLayer.weightsToNext[i];
        }
        // copy second part of the weights for layer from other network
        Layer otherLayer = other.layers.get(whichLayer);
        for (int i = whichNode; i < originalLayer.weightsToNext.length; i++) {
            mixLayer.weightsToNext[i] = otherLayer.weightsToNext[i];
        }
        newLayers.add(mixLayer);

        // copy all layers after the split one from other network
        for (int i = whichLayer + 1; i < other.layers.size(); i++) {
            newLayers.add(other.layers.get(i).clone());
        }

        nextGenerationNetwork.layers = newLayers;
        return nextGenerationNetwork;
    }

    public void mutate(Network network, double mutationRate) {
        if (mutationRate >= 1.0) {
            throw new IllegalArgumentException("Mutation rate should be strictly less than 1");
        }
        for (int layerIndex = 0; layerIndex < network.layers.size() - 1; layerIndex++) {
            Layer layer = network.layers.get(layerIndex);
            for (int i = 0; i < layer.getNbNodes(); i++) {
                for (int j = 0; j < network.layers.get(layerIndex + 1).getNbNodes(); j++) {
                    if (random.nextDouble() < mutationRate) {
                        layer.weightsToNext[i][j] += random.nextGaussian() / 5;

                        if (layer.weightsToNext[i][j] > 1) {
                            layer.weightsToNext[i][j] = 1;
                        }
                        if (layer.weightsToNext[i][j] < -1) {
                            layer.weightsToNext[i][j] = -1;
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

    int getRandomNodeForCrossover(int whichLayer, int nbNodes) {
        return random.nextInt(nbNodes);
    }
}

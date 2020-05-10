package net.mnementh64.neural;

import net.mnementh64.neural.model.layer.Layer;
import net.mnementh64.neural.model.weight.WeightUtils;
import org.junit.Assert;
import org.junit.Test;

public class NetworkGeneticEvolutionTest {

    @Test
    public void SPBX() throws Exception {
        // Given
        NetworkGeneticEvolution geneticEvolution = new NetworkGeneticEvolution() {
            @Override
            int getRandomLayerForCrossover(int nbLayers) {
                return 1;
            }
        };
        Network network1 = createNetwork1();
        Network network2 = createNetwork2();

        // When
        Network nextGenerationNetwork = geneticEvolution.SPBX(network1, network2);

        // Then
        Assert.assertEquals(4, nextGenerationNetwork.layers.size());
        Assert.assertEquals(Layer.Type.INPUT, nextGenerationNetwork.layers.get(0).type);
        Assert.assertEquals(10, nextGenerationNetwork.layers.get(0).getNbNodes());

        Assert.assertEquals(Layer.Type.HIDDEN, nextGenerationNetwork.layers.get(1).type);
        Assert.assertEquals(20, nextGenerationNetwork.layers.get(1).getNbNodes());
        Assert.assertEquals(1.0, nextGenerationNetwork.layers.get(1).weightsToNext[0][1], 0.0);
        Assert.assertEquals(5001.0, nextGenerationNetwork.layers.get(1).weightsToNext[5][1], 0.0);
        Assert.assertEquals(6001.0, nextGenerationNetwork.layers.get(1).weightsToNext[6][1], 0.0);
        Assert.assertEquals(19001.0, nextGenerationNetwork.layers.get(1).weightsToNext[19][1], 0.0);

        Assert.assertEquals(Layer.Type.HIDDEN, nextGenerationNetwork.layers.get(2).type);
        Assert.assertEquals(18, nextGenerationNetwork.layers.get(2).getNbNodes());
        // all weights from network 1
        Assert.assertEquals(1.0, nextGenerationNetwork.layers.get(2).weightsToNext[0][1], 0.0);
        Assert.assertEquals(1001.0, nextGenerationNetwork.layers.get(2).weightsToNext[5][1], 0.0);
        Assert.assertEquals(1201.0, nextGenerationNetwork.layers.get(2).weightsToNext[6][1], 0.0);
        Assert.assertEquals(3401.0, nextGenerationNetwork.layers.get(2).weightsToNext[17][1], 0.0);

        Assert.assertEquals(Layer.Type.OUTPUT, nextGenerationNetwork.layers.get(3).type);
        Assert.assertEquals(4, nextGenerationNetwork.layers.get(3).getNbNodes());
    }

    @Test
    public void mutate() throws Exception {
        // Given
        NetworkGeneticEvolution geneticEvolution = new NetworkGeneticEvolution();
        Network network = createNetwork1();

        // When
        geneticEvolution.mutate(network, 0.5, true, -1, 1);

        // Then
        // must have more weights with value 1 than before mutation (because max value is 1 once mutated and hidden weight mostly have values greater than 1)
        int count = 0;
        for (int layerIndex = 0; layerIndex < network.layers.size() - 1; layerIndex++) {
            Layer layer = network.layers.get(layerIndex);
            for (int i = 0; i < layer.getNbNodes(); i++) {
                for (int j = 0; j < network.layers.get(layerIndex + 1).getNbNodes(); j++) {
                    if (layer.weightsToNext[i][j] == 1.0) {
                        count++;
                    }
                    ;
                }
            }
        }
        System.out.println("count = " + count);
        Assert.assertTrue(count > 2);
    }

    private Network createNetwork1() throws Exception {
        Network network1 = new Network.Builder()
                .setWeightInitFunction(WeightUtils.gaussianNormalizedFunction)
                .addLayer(10)
                .addLayer(20)
                .addLayer(18)
                .addLayer(4)
                .build();
        // force weight values
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 18; j++) {
                network1.layers.get(1).weightsToNext[i][j] = i * 100 + j;
            }
        }
        for (int i = 0; i < 18; i++) {
            for (int j = 0; j < 4; j++) {
                network1.layers.get(2).weightsToNext[i][j] = i * 200 + j;
            }
        }
        return network1;
    }

    private Network createNetwork2() throws Exception {
        Network network2 = new Network.Builder()
                .setWeightInitFunction(WeightUtils.unitFunction)
                .addLayer(10)
                .addLayer(20)
                .addLayer(18)
                .addLayer(4)
                .build();
        // force weight values
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 18; j++) {
                network2.layers.get(1).weightsToNext[i][j] = i * 1000 + j;
            }
        }
        for (int i = 0; i < 18; i++) {
            for (int j = 0; j < 4; j++) {
                network2.layers.get(2).weightsToNext[i][j] = 10000 + i * 2000 + j;
            }
        }
        return network2;
    }

    @Test
    public void checkSimilarity() throws Exception {
        // Given
        NetworkGeneticEvolution geneticEvolution = new NetworkGeneticEvolution();
        Network network1 = createNetwork1();
        Network network2 = createNetwork2();

        // When
        geneticEvolution.checkSimilarity(network1, network2);

        // Then
        // no exception ? test is successful
    }
}
package net.mnementh64.neural.model.layer;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import net.mnementh64.neural.model.Node;
import net.mnementh64.neural.model.activation.ActivationFunction;
import net.mnementh64.neural.model.weight.WeightInitFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.EXISTING_PROPERTY, property = "type")
@JsonSubTypes(
        {
                @JsonSubTypes.Type(name = "INPUT", value = InputLayer.class),
                @JsonSubTypes.Type(name = "HIDDEN", value = HiddenLayer.class),
                @JsonSubTypes.Type(name = "OUTPUT", value = OutputLayer.class),
                @JsonSubTypes.Type(name = "RECURRENT", value = RecurrentLayer.class),
        })
@JsonInclude(JsonInclude.Include.NON_NULL)
public abstract class Layer {

    @JsonProperty
    public Type type;
    /**
     * Activation function of all nodes : sigmoide, hyperbolic tangent, ...
     */
    @JsonProperty
    public ActivationFunction activationFunction;
    /**
     * Weight initialization function
     */
    @JsonProperty
    public WeightInitFunction weightInitFunction;
    /**
     * weights between layer nodes and next layer nodes
     */
    @JsonProperty
    public double[][] weightsToNext;
    /**
     * ordered collection of nodes
     */
    @JsonIgnore
    List<Node> nodes;

    Layer(Type type) {
        this.type = type;
    }

    Layer(Type type, ActivationFunction activationFunction, int nbNodes) {
        this.type = type;
        this.activationFunction = activationFunction;

        // create nodes
        setNbNodes(nbNodes);
    }

    public abstract Layer clone();

    protected void cloneProperties(Layer clone) {
        clone.activationFunction = this.activationFunction;
        clone.weightInitFunction = this.weightInitFunction;
        if (this.weightsToNext != null) {
            clone.weightsToNext = new double[this.weightsToNext.length][];
            for (int i = 0; i < this.weightsToNext.length; i++) {
                clone.weightsToNext[i] = new double[this.weightsToNext[i].length];
                for (int j = 0; j < this.weightsToNext[i].length; j++) {
                    clone.weightsToNext[i][j] = this.weightsToNext[i][j];
                }
            }
        }
        clone.setNbNodes(this.getNbNodes());
    }

    public void linkTo(Layer nextLayer, WeightInitFunction weightInitFunction) throws Exception {
        this.weightInitFunction = weightInitFunction;
        this.weightsToNext = weightInitFunction.init(this.nodes.size(), nextLayer.nodes.size());
    }

    public void reset() {
        this.weightsToNext = weightInitFunction.init(weightsToNext.length, weightsToNext[0].length);
    }

    public void clearNodes() {
        this.nodes.forEach(node -> {
            node.input = 0;
            node.value = 0;
            node.delta = 0;
        });
    }

    public void init(List<Double> input) throws Exception {
        if (input.size() != getNbNodes())
            throw new Exception("Input values are bad sized for this layer : get " + input.size() + " items and expected " + getNbNodes());

        IntStream.range(0, input.size())
                .forEach(i -> nodes.get(i).value = input.get(i));
    }

    public void feedForward(Layer previousLayer) throws Exception {
        // compute each current layer's node's value
        for (int j = 0; j < nodes.size(); j++) {
            Node node = nodes.get(j);
            node.input = previousLayer.computeOutputToNode(j);
            node.value = activationFunction.apply(node.input);
        }
    }

    public void computeDelta(Layer nextLayer) throws Exception {
        // compute each current layer's node's delta
        for (int j = 0; j < nodes.size(); j++) {
            Node node = nodes.get(j);
            double delta = nextLayer.computeWeightedDelta(weightsToNext[j]);
            node.delta = delta * activationFunction.applyDerivative(node.input);
        }
    }

    public void adjustWeights(Layer nextLayer, double learningRate) throws Exception {
        for (int i = 0; i < nodes.size(); i++) {
            Node node = nodes.get(i);

            for (int j = 0; j < nextLayer.getNbNodes(); j++) {
                double nextDelta = nextLayer.getDelta(j);
                weightsToNext[i][j] += learningRate * node.value * nextDelta;
            }
        }
    }

    public List<Double> output() {
        return nodes.stream()
                .map(n -> n.value)
                .collect(Collectors.toList());
    }

    public int getNbNodes() {
        return nodes.size();
    }

    public void setNbNodes(int nbNodes) {
        nodes = new ArrayList<>(nbNodes);
        IntStream.rangeClosed(1, nbNodes)
                .forEach(i -> nodes.add(new Node()));
    }

    double getDelta(int j) throws Exception {
        return nodes.get(j).delta;
    }

    double computeWeightedDelta(double[] weights) throws Exception {
        double value = 0;
        for (int i = 0; i < nodes.size(); i++)
            value += weights[i] * nodes.get(i).delta;

        return value;
    }

    double computeOutputToNode(int nextLayerNodeIndex) throws Exception {
        double value = 0;

        for (int i = 0; i < nodes.size(); i++)
            value += nodes.get(i).value * weightsToNext[i][nextLayerNodeIndex];

        return value;
    }

    public enum Type {
        INPUT, HIDDEN, OUTPUT, RECURRENT
    }
}

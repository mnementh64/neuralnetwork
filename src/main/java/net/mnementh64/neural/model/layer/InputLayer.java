package net.mnementh64.neural.model.layer;

public class InputLayer extends Layer {

    public InputLayer() {
        super(Type.INPUT);
    }

    public InputLayer(int nbNodes) {
        super(Type.INPUT, null, nbNodes);
    }

    @Override
    public void feedForward(Layer previousLayer) throws Exception {
        // do nothing for first layer
    }

    @Override
    public Layer clone() {
        Layer clone = new InputLayer();
        super.cloneProperties(clone);

        return clone;
    }
}

package net.mnementh64.neural.model.activation;

public class ReLuFunction extends ActivationFunction {

    public ReLuFunction() {
        super(Type.RELU);
    }

    @Override
    public double apply(double x) {
        return Math.max(0, x);
    }

    @Override
    public double applyDerivative(double x) {
        return x <= 0 ? 0 : 1;
    }
}

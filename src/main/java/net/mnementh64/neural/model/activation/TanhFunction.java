package net.mnementh64.neural.model.activation;

public class TanhFunction extends ActivationFunction {

    public TanhFunction() {
        super(Type.TANH);
    }

    @Override
    public double apply(double x) {
        return Math.tanh(x);
    }

    @Override
    public double applyDerivative(double x) {
        return 1 / (Math.cosh(x) * Math.cosh(x));
    }
}

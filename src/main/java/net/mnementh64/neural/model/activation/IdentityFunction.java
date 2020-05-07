package net.mnementh64.neural.model.activation;

public class IdentityFunction extends ActivationFunction {

    public IdentityFunction() {
        super(Type.IDENTITY);
    }

    @Override
    public double apply(double x) {
        return x;
    }

    @Override
    public double applyDerivative(double x) {
        return 1.0;
    }
}

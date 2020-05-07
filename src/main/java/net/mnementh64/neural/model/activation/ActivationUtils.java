package net.mnementh64.neural.model.activation;

public class ActivationUtils {

    public static final ActivationFunction identity = new IdentityFunction();
    public static final ActivationFunction sigmoid = new SigmoideFunction();
    public static final ActivationFunction tanh = new TanhFunction();
    public static final ActivationFunction relu = new ReLuFunction();
}

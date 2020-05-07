package net.mnementh64.neural.model.weight;

public class UnitFunction extends WeightInitFunction {

    double value;

    public UnitFunction() {
    }

    public UnitFunction(double value) {
        super(Type.UNIT);
        this.value = value;
    }

    @Override
    public double[][] init(int s1, int s2) {
        double[][] weights = new double[s1][s2];
        for (int i1 = 0; i1 < s1; i1++)
            for (int i2 = 0; i2 < s2; i2++)
                weights[i1][i2] = value;
        return weights;
    }
}

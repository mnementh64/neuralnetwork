package net.mnementh64.neural.model.weight;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Random;

public class RandomUniformFunction extends WeightInitFunction {

    @JsonProperty
    double min;
    @JsonProperty
    double max;

    public RandomUniformFunction() {
        super(Type.RANDOM_UNIFORM);
    }

    public RandomUniformFunction(double min, double max) {
        super(Type.RANDOM_UNIFORM);
        this.min = min;
        this.max = max;
    }

    @Override
    public double[][] init(int s1, int s2) {
        double[][] weights = new double[s1][s2];
        Random random = new Random();
        random.setSeed(System.currentTimeMillis());
        for (int i1 = 0; i1 < s1; i1++)
            for (int i2 = 0; i2 < s2; i2++)
                weights[i1][i2] = (max - min) * random.nextDouble() + min;

        return weights;
    }
}

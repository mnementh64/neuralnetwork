package net.mnementh64.neural.model;

import java.util.List;

public class DataRow {

    public List<Double> input;
    public List<Double> expectedOutput;

    public DataRow(List<Double> input, List<Double> expectedOutput) {
        this.input = input;
        this.expectedOutput = expectedOutput;
    }
}

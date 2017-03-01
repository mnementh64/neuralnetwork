package net.mnementh64.neural.model;

import java.util.List;

public class DataRow
{

	public List<Float> input;
	public List<Float> expectedOutput;

	public DataRow(List<Float> input, List<Float> expectedOutput)
	{
		this.input = input;
		this.expectedOutput = expectedOutput;
	}
}

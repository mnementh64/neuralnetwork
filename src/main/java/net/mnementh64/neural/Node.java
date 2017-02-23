package net.mnementh64.neural;

public class Node
{

	/**
	 * input value before activation function application
	 */
	public float input;
	/**
	 * node value after activation function application
	 */
	public float value;
	/**
	 * error retropropagation value at this node
	 */
	public float delta;
}

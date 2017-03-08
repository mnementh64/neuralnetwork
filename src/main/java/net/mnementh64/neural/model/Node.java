package net.mnementh64.neural.model;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class Node
{

	/**
	 * input value before activation function application
	 */
	@JsonIgnore
	public double input;
	/**
	 * node value after activation function application
	 */
	public double value;
	/**
	 * error retropropagation value at this node
	 */
	@JsonIgnore
	public double delta;

}

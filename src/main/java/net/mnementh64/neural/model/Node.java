package net.mnementh64.neural.model;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class Node
{

	/**
	 * input value before activation function application
	 */
	@JsonIgnore
	public float input;
	/**
	 * node value after activation function application
	 */
	public float value;
	/**
	 * error retropropagation value at this node
	 */
	@JsonIgnore
	public float delta;

}

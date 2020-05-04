package net.mnementh64.neural.model.weight;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.EXISTING_PROPERTY, property = "type", visible = true)
@JsonSubTypes(
{
		@JsonSubTypes.Type(name = "RANDOM_UNIFORM", value = RandomUniformFunction.class),
		@JsonSubTypes.Type(name = "RANDOM_GAUSSIAN", value = RandomGaussianFunction.class),
		@JsonSubTypes.Type(name = "UNIT", value = UnitFunction.class)
})
public abstract class WeightInitFunction
{

	enum Type
	{
		RANDOM_UNIFORM, RANDOM_GAUSSIAN, UNIT
	}

	@JsonProperty
	Type type;

	protected WeightInitFunction()
	{

	}

	protected WeightInitFunction(Type type)
	{
		this.type = type;
	}

	/**
	 * Compute the initial weights of all links between nodes of 2 layers
	 * @param s1 : current layer size
	 * @param s2 : next layer size
	 * @return the weight value for links between all nodes of current layer and all nodes of next layer.
	 */
	public abstract double[][] init(int s1, int s2);
}

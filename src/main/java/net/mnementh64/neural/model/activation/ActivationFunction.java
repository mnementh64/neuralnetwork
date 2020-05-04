package net.mnementh64.neural.model.activation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.EXISTING_PROPERTY, property = "type", visible = true)
@JsonSubTypes(
{
		@JsonSubTypes.Type(name = "IDENTITY", value = IdentityFunction.class),
		@JsonSubTypes.Type(name = "SIGMOID", value = SigmoideFunction.class),
		@JsonSubTypes.Type(name = "TANH", value = TanhFunction.class),
		@JsonSubTypes.Type(name = "RELU", value = ReLuFunction.class)
})
public abstract class ActivationFunction
{

	enum Type
	{
		IDENTITY, SIGMOID, TANH, RELU
	}

	@JsonProperty
	private Type type;

	protected ActivationFunction()
	{

	}

	protected ActivationFunction(Type type)
	{
		this.type = type;
	}

	public abstract double apply(double x);

	public abstract double applyDerivative(double x);
}

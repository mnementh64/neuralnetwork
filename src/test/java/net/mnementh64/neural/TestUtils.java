package net.mnementh64.neural;

import java.io.InputStream;

class TestUtils
{

	static String loadResource(String resourceName) throws Exception
	{
		ClassLoader classloader = Thread.currentThread().getContextClassLoader();
		try (InputStream is = classloader.getResourceAsStream(resourceName))
		{
			byte[] values = new byte[is.available()];
			is.read(values);
			return new String(values);
		}
	}

}

package cs12b.mySQL_PA;

public class SQLParser {

	public SQLParser(Database db) {
		
	}
	
	public void parse(String sql) {
		
	}
	
	/**
	 * This method will split a given String into an array of characters 
	 * @param str the input String
	 * @return a String array of each character in str
	 */
	public String[] splitStringIntoChars(String str) {
		return str.split("");
	}
	
	/**
	 * This method will extract a given String from brackets [ ] if it's found inside them 
	 * @param str the String to extract
	 * @return str without its brackets
	 * @throws IllegalArgumentException if str does not have brackets
	 */
	public String extractStringFromBrackets(String str) {
		try {
			return str.substring(str.indexOf("[") + 1, str.indexOf("]"));
		} catch (Exception IndexOutOfBoundsException){
			throw new IllegalArgumentException("expected [...] got " + str);
		}
	}
	
	/**
	 * This method will extract a given String from parens ( ) if it's found inside them 
	 * @param str the String to extract
	 * @return a String array of string arguments extracted from parens
	 * @throws IllegalArgumentException if str does not have parens
	 */
	public static String[] extractStringsFromParens(String str) {
		try {
			return str.substring(str.indexOf("(") + 1, str.indexOf(")")).split(", ");
		} catch (Exception IndexOutOfBoundsException){
			throw new IllegalArgumentException("expected (...) got " + str);
		}
	}
	
	/**
	 * Retrieves the String representation of the "order by" alias from the given String 
	 * @param str the given String 
	 * @return a String representing the order by alias
	 */
	public String extractOrderByAlias(String str) {
		try {
			return extractStringsFromParens(str.substring(str.toUpperCase().indexOf("ORDER BY")))[0];
		} catch (Exception IndexOutOfBoundsException){
			return null;
		}
	}
}

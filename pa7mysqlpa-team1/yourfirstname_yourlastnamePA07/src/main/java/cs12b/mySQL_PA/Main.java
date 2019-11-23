package cs12b.mySQL_PA;

import java.util.Scanner;

/**
 * Main class that runs the overall SQL program. 
 * <p>
 * You <b>should not</b> edit this file. All your code should be in SQLParser and Database. 
 * </p>
 * 
 * @author Eitan Joseph
 * @version 1.0 
 * COSI12B: mySQL PA 
 * 11/15/2019
 */
public class Main {

	/**
	 * The sentinel the user should enter to exit the program 
	 */
	public static final String SENTINEL = "Q";
	
	/**
	 * The main method: runs the program.
	 * @param args the program arguments (not used) 
	 */
	public static void main(String[] args) {
		SQLParser parser = new SQLParser(new Database()); // construct SQL parser 
		Scanner consoleRdr = new Scanner(System.in);
		boolean running = true;
		String line = "";
		do {
			// get line from user input, if it's the sentinel => exit
			// otherwise, parse the line using the parser 
			line = consoleRdr.nextLine();
			if (line.equalsIgnoreCase(SENTINEL)) {
				running = false;
			}
			else {
				parser.parse(line);
			}
		} while (running);
		
		consoleRdr.close();
	}
}

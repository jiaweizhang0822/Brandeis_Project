package cs12b.mySQL_PA;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.StringJoiner;

/**
 * This class provides the ability to test some generic program that uses console input and output.
 * It has replaced the old testing system which comprised of utility files full of static methods that
 * had to be tweaked for each assignment.
 * 
 * @author Eitan Joseph, Chami Lamelas
 * @version 2.0 
 */
public class GenericConsoleTester {
	
	/**
	 * Gets the line separator used by the O.S.
	 */
	public static final String SYS_LINE_SEPARATOR = System.getProperty("line.separator");
	/**
	 * Will be using the UTF-8 character set 
	 */
	public static final Charset CHAR_SET = StandardCharsets.UTF_8;
	
	/**
	 * The old input stream 
	 */
	private InputStream oldIn;
	/**
	 * The old output stream 
	 */
	private PrintStream oldOut;
	/**
	 * The new output stream (used by various methods). The new input stream is used once and is not
	 * stored as a field. 
	 */
	private PrintStream newPrintOut;
	/**
	 * The new output stream (used only in default output generation testing). 
	 */
	private ByteArrayOutputStream byteOut;
	/**
	 * If generating files, keep a list of files. 
	 */
	private List<String> outFilePaths;
	/**
	 * Determines whether or not user wishes to keep generated files.
	 */
	private boolean keepOutFiles;
	
	/**
	 * Constructs a default tester. That is, one that does not generate output files in the testing process. 
	 */
	public GenericConsoleTester() {
		oldIn = null;
		oldOut = null;
		newPrintOut = null;
		byteOut = null;
		outFilePaths = null;
		keepOutFiles = false;
	}
	
	/**
	 * Constructs a tester that generates output files for testing. One can specify whether or not to keep them.
	 * @param keepOutFiles boolean representing whether or not to keep output files
	 */
	public GenericConsoleTester(boolean keepOutFiles) {
		this();
		outFilePaths = new LinkedList<String>();
		this.keepOutFiles = keepOutFiles;
	}

	/**
	 * Backs up the default system input/output streams. 
	 */
	public void storeOldStreams() {
		oldIn = System.in;
		oldOut = System.out;
	}

	/**
	 * Resets the default system input/output streams and deletes any opened files.
	 */
	public void cleanUpStreamsAndFiles() {
		if (oldIn == null || oldOut == null) {
			throw new NoSuchElementException("Cannot restore system streams.");
		}
		
		System.setIn(oldIn);
		System.setOut(oldOut);
		
		if (outFilePaths != null && !keepOutFiles) {
			for (String filePath : outFilePaths) {
				File file = new File(filePath);
				if (file.exists()) {
					file.delete();
				}
				else {
					System.err.println(file.getAbsolutePath() + " could not be deleted. Did you already delete it?");
				}
			}
		}
	}
	
	/**
	 * Sets up the input stream given an arbitrary number of Strings (or String array). It is assumed that the elements in the input parameter are
	 * to be separate lines of input in the test program. 
	 * @param linesToTest lines of input to test
	 */
	public void setUpInputStream(String... linesToTest) {
		StringBuilder inBuilder = new StringBuilder();
		for (String line : linesToTest) {
			inBuilder.append(line);
			inBuilder.append(SYS_LINE_SEPARATOR);
		}
		ByteArrayInputStream newInStream = new ByteArrayInputStream(inBuilder.toString().getBytes(CHAR_SET));
		System.setIn(newInStream);
	}
	
	/**
	 * Sets up the output stream for default testing; that is, with no output file generation. 
	 * @throws IllegalArgumentException if file-generation testing was chosen in object construction
	 */
	public void setUpOutStream() { 
		if (outFilePaths != null) {
			throw new IllegalArgumentException("File name must be specified.");
		}
		byteOut = new ByteArrayOutputStream();
		newPrintOut = new PrintStream(byteOut);
		System.setOut(newPrintOut);
	}
	
	/**
	 * Sets up the output stream to a specified file. This is to be used in the file-generation version of testing.
	 * @param filePath the file path to the generated output file 
	 * @throws IllegalArgumentException if default testing was chosen in object construction
	 */
	public void setUpOutStream(String filePath) {
		if (outFilePaths == null) {
			throw new IllegalArgumentException("Not generating files");
		}
		File file = new File(filePath);
		newPrintOut = null;
		try {
			newPrintOut = new PrintStream(file);
			System.setOut(newPrintOut);
			outFilePaths.add(filePath);
		}
		catch (FileNotFoundException f) {
			System.err.println("Could not create output file: " + f.getMessage());
		}
	}

	/**
	 * Gets the actual output for the test. That is, it gets the student's generated output (regardless of the output method) after their code has
	 * been run. 
	 * @return the actual output for the test or null if an error occurred in processing the output   
	 * @throws NoSuchElementException if output stream was not created / 
	 */
	public String getActual() {
		if (newPrintOut == null) {
			throw new NoSuchElementException("Output print stream not created.");
		}
		newPrintOut.close();
		if (outFilePaths != null) {
			if (outFilePaths.isEmpty()) {
				throw new NoSuchElementException("No output file created from which to read output.");
			}
			File actualFile = new File(outFilePaths.get(outFilePaths.size() - 1));
			Scanner actualRdr = null;
			StringJoiner joiner = new StringJoiner("\n");
			try {
				actualRdr = new Scanner(actualFile);
				while (actualRdr.hasNextLine()) {
					joiner.add(actualRdr.nextLine().trim());
				}
				return joiner.toString();
			}
			catch (FileNotFoundException f) {
				System.err.println("Could not read output file: " + f.getMessage());
			}
			finally {
				if (actualRdr != null) {
					actualRdr.close();
				}
			}
			return null;
		}
		else { 
			try {
				byteOut.close();
			}
			catch (IOException e) {
				System.err.println(e.getMessage());
			}
			try {
				String out = byteOut.toString(StandardCharsets.UTF_8.name()).trim();
				StringJoiner joiner = new StringJoiner("\n");
				String[] splitOnLineSep = out.split(SYS_LINE_SEPARATOR);
				for (String separated : splitOnLineSep) {
					joiner.add(separated.trim());
				}
				return joiner.toString();
			} catch (UnsupportedEncodingException e) {
				System.err.println(e.getMessage());
				return null;
			}
		}
	}
}

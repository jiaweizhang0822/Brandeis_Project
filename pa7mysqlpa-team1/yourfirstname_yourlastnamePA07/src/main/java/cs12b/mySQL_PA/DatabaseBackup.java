package cs12b.mySQL_PA;

import java.util.NoSuchElementException;

/**
 * This class provides a data structure that allows you to back up databases. You can back up a maximum of 3
 * databases. The underlying data structure is a doubly linked list based stack.
 * <p>
 * You <b>should not</b> edit this class. However, feel free to study its implementation.
 * </p>
 * 
 * @author Chami Lamelas
 * @version 2.0
 * COSI 12B: MySQL PA
 * 11/15/2019
 */
public class DatabaseBackup {
	
	/*
	 * Doubly linked list node to be used in DLL-stack based implementation of Database backup 
	 */
	private static class Node {

		/**
		 * Data of the node: database 
		 */
		private Database db;
		
		/**
		 * Reference to the next node in the structure
		 */
		private Node next;
		
		/**
		 * Reference to the previous node in the structure
		 */
		private Node prev;
		
		/**
		 * Constructs a node containing a given Database
		 * @param db a database 
		 */
		public Node(Database db) {
			this.db = db;
			next = null;
			prev = null;
		}
		
		/**
		 * Updates the next node reference  
		 * @param next new next node reference 
		 */
		public void setNext(Node next) {
			this.next = next;
		}
		
		/**
		 * Gets the next node reference stored in this node 
		 * @return next node reference of this node
		 */
		public Node getNext() {
			return next;
		}
		
		/**
		 * Updates the previous node reference  
		 * @param prev new previous node reference 
		 */
		public void setPrev(Node prev) {
			this.prev = prev;
		}
		
		/**
		 * Gets the previous node reference stored in this node 
		 * @return previous node reference of this node
		 */
		public Node getPrev() {
			return prev;
		}
		
		/**
		 * Gets the database in the node 
		 * @return the database in the node
		 */
		public Database getDatabase() {
			return db;
		}
	}

	/**
	 * Maximum number of databases that can be backed up
	 */
	private static final int LIMIT = 3;
	
	/**
	 * Head of the underlying DLL-based stack. This will store the oldest backup.
	 */
	private Node head;
	
	/**
	 * Tail of the underlying DLL-based stack. This will store the newest backup. 
	 */
	private Node tail;
	
	/**
	 * Number of Databases that have been currently backed up
	 */
	private int numDbs;
	
	/**
	 * Initializes a database backup structure.
	 */
	public DatabaseBackup() {
		head = null;
		tail = null;
		numDbs = 0;
	}
	
	/**
	 * Backs up a database in the structure. If it exceeds the maximum capacity it will delete the oldest 
	 * backup automatically. 
	 * @param db a database to back up 
	 */
	public void backUp(Database db) {
		// if no. of dbs has hit the limit, delete the oldest db (stored at head) 
		if (numDbs == LIMIT) {
			head.getNext().setPrev(null);
			head = head.getNext();
			numDbs--; // track deletion
		}
		Node newNode = new Node(db);
		// Edge case: empty structure, set head and tail to be new Node 
		if (numDbs == 0) {
			head = newNode;
			tail = head;
		}
		// In other cases, add at tail so that tail is updated to be the most recent backup 
		else {
			tail.setNext(newNode);
			newNode.setPrev(tail);
			tail = newNode;
		}
		numDbs++; // track addition
	}
	
	/**
	 * Gets the most recent backup from the structure and removes it. This means between 0 and 2 Databases
	 * remain backed up.
	 * @return the latest backup 
	 */
	public Database getLatestBackUp() {
		if (isEmpty()) { 
			throw new NoSuchElementException("No backups to retrieve.");
		}
		Database db = tail.getDatabase(); // to store most recent back up
		if (numDbs == 1) { // if only 1 db to remove, set both head, tail to null 
			head = null;
			tail = null;
		}
		else { // move tail backwards to delete most recent back up
			tail.getPrev().setNext(null);
			tail = tail.getPrev();
		}
		numDbs--; // decrease no. of dbs 
		return db;
	}

	/**
	 * Checks if the database structure has any more databases backed up. 
	 * @return if there are no dbs backed up
	 */
	public boolean isEmpty() {
		return numDbs==0;
	}
}

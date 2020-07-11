/*
 * Peersim-Gadget : A Gadget protocol implementation in peersim based on the paper
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 * Copyright (C) 2012
 * Deepak Nayak 
 * Columbia University, Computer Science MS'13
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */
// todo: replace run directory names with meaningful ones
// todo: copy the config file used inside the run directory
// todo: clean up config files
// todo: separate the writing into files from the training method

package peersim.gossip;
import com.google.gson.JsonObject;
import peersim.config.*;
import peersim.core.*;


/**
 * Class PegasosNode
 * An implementation of {@link Node} which can handle external resources. 
 * It is based on {@link GeneralNode} and extended it to be used with pegasos solver.
 * At the start of node, each node has some associated data file, on which it calls
 * training function through a jni call. 
 * It will take the resourcepath from config file.
 * This is another implementation of {@link Node} class that is used to compose the
 * p2p {@link Network}, where each node can handle an external resource.
 * @author Nitin Nataraj
 */
public class PegasosNode implements Node {

	// ================= fields ========================================
	// =================================================================

	
	// Neural net params in the config file
	private static final String PAR_LEARNING_RATE = "learning_rate";
	private static final String PAR_BATCH_SIZE = "batch_size";
	private static final String PAR_INITIALIZATION = "init_method";
	private static final String PAR_PATH = "resourcepath";	
	private static final String PAR_NUMHIDDEN_1 = "numhidden_1";
	private static final String PAR_NUMHIDDEN_2 = "numhidden_2";
	private static final String PAR_NUMHIDDEN_3 = "numhidden_3";
	private static final String PAR_CYCLES_FOR_CONVERGENCE = "cycles_for_convergence";
	private static final String PAR_CONVERGENCE_EPSILON = "convergence_epsilon";
	private static final String PAR_RANDOM_SEED = "random.seed";
	private static final String PAR_LOSS_FUNCTION = "loss_function";
	private static final String PAR_HIDDEN_ACT = "hidden_layer_act";
	private static final String PAR_FINAL_ACT = "final_layer_act";
	private static final String PAR_FEATURE_SPLIT_TYPE = "feature_split_type";
	private static final String PAR_OVERLAP_RATIO = "overlap_ratio";
	private static final String PAR_NN_TYPE = "nn_type";
	private static final String PAR_NUM_LAYERS = "num_layers";
	private static final String PAR_RUN = "run";
	private static final String PAR_DATASET_NAME = "dataset_name";
	private static final String PAR_NODE_ID = "node_id";
	private static final String PAR_NUM_NODES = "num_nodes";
	private static final String PAR_RUN_TYPE = "run_type";
	private static final String PAR_NEIGHBOR = "neighbor";
	

	
	private static long counterID = -1; // used to generate unique IDs 
	protected Protocol[] protocol = null; //The protocols on this node.
	
 	//learning parameters of the Neural Network 
	public double learning_rate;
	public int epochs;
	public int batch_size;
	public int initmethod; // Method to initialize the weights of the NN. 0 - Random, 1 - Xavier
	public int activationmethod; // Activation function of layers of NN. 0 - Identity, 1 - Sigmoid, 2 - RELU, 3 - Tanh 
	/**
	 * The current index of this node in the node
	 * list of the {@link Network}. It can change any time.
	 * This is necessary to allow
	 * the implementation of efficient graph algorithms.
	 */
	private int index;

	/**
	 * The fail state of the node.
	 */
	protected int failstate = Fallible.OK;

	/**
	 * The ID of the node. It should be final, however it can't be final because
	 * clone must be able to set it.
	 */
	public long ID;

	/**
	 * The prefix for the resources file. All the resources file will be in prefix 
	 * directory. later it should be taken from configuration file.
	 */
	public String resourcepath;
	private int num_nodes;
	public int num_run;
	public long num_features;
	public int num_classes;
	public int num_hidden_nodes_1;
	public int num_hidden_nodes_2;
	public int num_hidden_nodes_3;
	public String csv_filename;
	public String csv_predictions_filename;
	public String weights_filename;
	public String loss_func;
	public String hidden_layer_activation;
	public String final_layer_activation;
	public String feature_split_type;
	public double overlap_ratio;

	public String nn_type;
	public int num_layers;
	String dataset_name;
	int node_id;
	String run_type;
	
	
	// Variables to maintain loss
	public double train_loss = -1;
	public double test_loss = -1;
	public double train_auc = -1;
	
	
	// Variables to determine convergence
	public boolean converged = false;
	public int num_converged_cycles = 0;
	public double convergence_epsilon;
	public int cycles_for_convergence;
	
	// Seed
	public long random_seed;
	public String result_dir;
	
	// NNConfig
	public JsonObject nnconfig = null;
	
	
	
	// ================ constructor and initialization =================
	// =================================================================
	/** Used to construct the prototype node. This class currently does not
	 * have specific configuration parameters and so the parameter
	 * <code>prefix</code> is not used. It reads the protocol components
	 * (components that have type {@value peersim.core.Node#PAR_PROT}) from
	 * the configuration.
	 */
	
	

	
	
	public PegasosNode(String prefix) {
		
		// Obtain configurations set in the config file
		String[] names = Configuration.getNames(PAR_PROT);
		learning_rate = Configuration.getDouble(PAR_LEARNING_RATE);
		batch_size = Configuration.getInt(PAR_BATCH_SIZE, 4);
		initmethod = Configuration.getInt(PAR_INITIALIZATION, 0);
		resourcepath = (String)Configuration.getString(PAR_PATH);
		num_hidden_nodes_1 = Configuration.getInt(PAR_NUMHIDDEN_1); // number of nodes in hidden layer 1
		num_hidden_nodes_2 = Configuration.getInt(PAR_NUMHIDDEN_2); // number of nodes in hidden layer 2
		num_hidden_nodes_3 = Configuration.getInt(PAR_NUMHIDDEN_3, 20); // number of nodes in hidden layer 3
		cycles_for_convergence = Configuration.getInt(PAR_CYCLES_FOR_CONVERGENCE); 
		convergence_epsilon = Configuration.getDouble(PAR_CONVERGENCE_EPSILON);
		random_seed = Configuration.getLong(PAR_RANDOM_SEED); // will be used for determining feature splits
		loss_func = (String)Configuration.getString(PAR_LOSS_FUNCTION); // Loss function 
		hidden_layer_activation = (String)Configuration.getString(PAR_HIDDEN_ACT); // hidden layer activation
		final_layer_activation = (String)Configuration.getString(PAR_FINAL_ACT); // final layer activation
		feature_split_type = (String)Configuration.getString(PAR_FEATURE_SPLIT_TYPE); // overlap or non-overlap
		overlap_ratio = Configuration.getDouble(PAR_OVERLAP_RATIO); // if feature_split_type is overlap then, this ratio is used
		nn_type = Configuration.getString(PAR_NN_TYPE); // mlp, cnn, rnn, etc
		num_layers = Configuration.getInt(PAR_NUM_LAYERS); // number of hidden layers
		num_run = Configuration.getInt(PAR_RUN); // number of hidden layers
		dataset_name = Configuration.getString(PAR_DATASET_NAME); // number of hidden layers
	
		
		CommonState.setNode(this);
		ID = nextID();
		protocol = new Protocol[names.length];
		for (int i=0; i < names.length; i++) {
			CommonState.setPid(i);
			Protocol p = (Protocol) 
					Configuration.getInstance(names[i]);
			protocol[i] = p; 
		}
	}
	
	/**
	 * Used to create actual Node by calling clone() on a prototype node. So, actually 
	 * a Node constructor is only called once to create a prototype node and after that
	 * all nodes are created by cloning it.
	 
	 */
	
	public Object clone() {
		
		/*
		 * Initializes the node and makes an API call to the flask server to setup a neural network.
		 * All NN configurations are managed through here - type of NN, number of layers, node_id, dataset_name,
		 * learning rate, epochs, batch_size, activation, loss
		 */
		PegasosNode result = null;
		
		try { 
			result=(PegasosNode)super.clone(); 
			}
		catch( CloneNotSupportedException e ) {} // never happens
		
		result.protocol = new Protocol[protocol.length];
		CommonState.setNode(result);
		result.ID = nextID();
		for(int i=0; i<protocol.length; ++i) {
			CommonState.setPid(i);
			result.protocol[i] = (Protocol)protocol[i].clone();
		}
		System.out.println("Network Size "+ Network.size());
		System.out.println("creating node with ID: " + result.getID());
		
		// Determine base dataset name
		//String[] temp_data = resourcepath.split("/");
		String base_dataset_name = resourcepath; //temp_data[temp_data.length - 2];
        System.out.println("Base Dataset name" + base_dataset_name);
		
        
        // Create base NN on the Python side using these configurations
        // Create a JSON object with configurations
        JsonObject nnconfig = new JsonObject();

        nnconfig.addProperty(PAR_LEARNING_RATE, learning_rate);
        nnconfig.addProperty(PAR_BATCH_SIZE, batch_size);
        nnconfig.addProperty(PAR_INITIALIZATION, initmethod);
        nnconfig.addProperty(PAR_PATH, resourcepath);
        nnconfig.addProperty(PAR_NUMHIDDEN_1, num_hidden_nodes_1);
        nnconfig.addProperty(PAR_NUMHIDDEN_2, num_hidden_nodes_2);
        nnconfig.addProperty(PAR_NUMHIDDEN_3, num_hidden_nodes_3);
        nnconfig.addProperty(PAR_CYCLES_FOR_CONVERGENCE, cycles_for_convergence);
        nnconfig.addProperty(PAR_CONVERGENCE_EPSILON, convergence_epsilon);
        nnconfig.addProperty(PAR_RANDOM_SEED, random_seed);
        nnconfig.addProperty(PAR_LOSS_FUNCTION, loss_func);
        nnconfig.addProperty(PAR_HIDDEN_ACT, hidden_layer_activation);
        nnconfig.addProperty(PAR_FINAL_ACT, final_layer_activation);
        nnconfig.addProperty(PAR_FEATURE_SPLIT_TYPE, feature_split_type);
        nnconfig.addProperty(PAR_OVERLAP_RATIO, overlap_ratio);
        nnconfig.addProperty(PAR_NN_TYPE, nn_type);
        nnconfig.addProperty(PAR_NUM_LAYERS, num_layers);
        nnconfig.addProperty(PAR_DATASET_NAME, dataset_name);
        nnconfig.addProperty(PAR_NODE_ID, result.getID());
        nnconfig.addProperty(PAR_NUM_NODES, Network.size());


        if (Network.size() == 1) {
        	nnconfig.addProperty(PAR_RUN_TYPE, "centralized");
        }
        else {
        	nnconfig.addProperty(PAR_RUN_TYPE, "distributed");
        }
        nnconfig.addProperty(PAR_RUN, num_run);
        // Neighbor will be changed in GadgetProtocol
        nnconfig.addProperty(PAR_NEIGHBOR, -1);
        result.nnconfig = nnconfig;
        
        System.out.println("JSON payload being sent:");
        System.out.println(nnconfig.toString());
        
        // Send 'clear' command to clear existing neural net cluster
        if (result.getID() == 0) {
        	HTTPSendDetailsAtOnce.sendRequest("vpnn", "clear", nnconfig);
        }
        // Send the 'init' http command to setup NNs
        HTTPSendDetailsAtOnce.sendRequest("vpnn", "init", nnconfig);
		return result;
		
	}

 
	/** returns the next unique ID */
	private long nextID() {

		return counterID++;
	}

	// =============== public methods ==================================
	// =================================================================


	public void setFailState(int failState) {

		// after a node is dead, all operations on it are errors by definition
		if(failstate==DEAD && failState!=DEAD) throw new IllegalStateException(
				"Cannot change fail state: node is already DEAD");
		switch(failState)
		{
		case OK:
			failstate=OK;
			break;
		case DEAD:
			//protocol = null;
			index = -1;
			failstate = DEAD;
			for(int i=0;i<protocol.length;++i)
				if(protocol[i] instanceof Cleanable)
					((Cleanable)protocol[i]).onKill();
			break;
		case DOWN:
			failstate = DOWN;
			break;
		default:
			throw new IllegalArgumentException(
					"failState="+failState);
		}
	}

	public int getFailState() { return failstate; }

	public boolean isUp() { return failstate==OK; }

	public Protocol getProtocol(int i) { return protocol[i]; }

	public int protocolSize() { return protocol.length; }

	public int getIndex() { return index; }

	public void setIndex(int index) { this.index = index; }
        
        
	/**
	 * Returns the ID of this node. The IDs are generated using a counter
	 * (i.e. they are not random).
	 */
	public long getID() { return ID; }

	public String toString() 
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("ID: "+ID+" index: "+index+"\n");
		for(int i=0; i<protocol.length; ++i)
		{
			buffer.append("protocol[" + i +"]=" + protocol[i] + "\n");
		}
		return buffer.toString();
	}

	/** Implemented as <code>(int)getID()</code>. */
	public int hashCode() { return (int)getID(); }

	

}
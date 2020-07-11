/*
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

package peersim.gossip;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.eval.ROCBinary;

import java.util.Arrays;
import java.lang.Integer;

import java.io.FileReader;

import java.io.LineNumberReader;
import peersim.gossip.PegasosNode;

import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;
import peersim.Simulator;
import peersim.cdsim.*;


import java.net.MalformedURLException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.text.ParseException;
import java.io.BufferedReader;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 *  @author Nitin Nataraj
 */


public class GadgetProtocol implements CDProtocol {
	public static boolean flag = false;
	public static int t = 0;
	public static boolean optimizationDone = false;	
	public double EPSILON_VAL = 0.01;
	protected int lid;
	protected double lambda;
	protected int T;
	public static double[][] optimalB;
	public static int end = 0;
	public static boolean pushsumobserverflag = false;
	public static final int CONVERGENCE_COUNT = 10;
	private String protocol;
	private String resourcepath;


	/**
	 * Default constructor for configurable objects.
	 */
	public GadgetProtocol(String prefix) {
		lid = FastConfig.getLinkable(CommonState.getPid());
		
		protocol = Configuration.getString(prefix + "." + "prot", "pushsum1");
		
	}

	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverRequest(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}
	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverResponse(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}

	/**
	 * Clone an existing instance. The clone is considered 
	 * new, so it cannot participate in the aggregation protocol.
	 */
	public Object clone() {
		GadgetProtocol gp = null;
		try { gp = (GadgetProtocol)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return gp;
	}
	
	

	protected List<Node> getPeers(Node node) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) {
			List<Node> l = new ArrayList<Node>(linkable.degree());			
			for(int i=0;i<linkable.degree();i++) {
				l.add(linkable.getNeighbor(i));
			}
			return l;
		}
		else
			return null;						
	}			

	public void nextCycle(Node node, int pid) {
		
		PegasosNode pn = (PegasosNode)node; // Initializes the Pegasos Node
		int cycles = Configuration.getInt("simulation.cycles");
		// Sending a 'train' command will run one round of feedforward and backpropation steps
		// HTTP response object will contain convergence information
		
		// Train until convergence. Convergence is determined by the Python part of the code
		// using an epsilon criterion.
		
		if (Network.size() == 1) {
			if(!pn.converged) {
				// Get response object and check for convergence, update pn.converged
				boolean response = HTTPSendDetailsAtOnce.sendRequest("vpnn", "train", pn.nnconfig);
				System.out.println("Response from GP: "+ response);
				pn.converged = response; // Update converged flag
				
				// Compute loss
				HTTPSendDetailsAtOnce.sendRequest("vpnn", "calc_losses", pn.nnconfig);
				
				
			}
			
			else {
				
				System.out.println("Node " + pn.getID() + " has converged.");
			}
			
		}
	
		else {
			
			
			// check if node has converged
			if(!pn.converged) {
				
				// Determine neighbor and update neighbor property of original node
				PegasosNode peer = (PegasosNode)selectNeighbor(node, pid);
				pn.nnconfig.addProperty("neighbor", peer.getID());
				
				// Send a command called gossip - which will use the current node ID, and neighbor ID
				// and do feedforward on both -  share losses and then backpropagate on both
				boolean response = HTTPSendDetailsAtOnce.sendRequest("vpnn", "gossip", pn.nnconfig);
				pn.converged = response;
				
				
			}
			else {
				System.out.println("Node " + pn.getID() + " has converged.");
				
			}
			if (pn.getID() == Network.size()-1) {
				
				HTTPSendDetailsAtOnce.sendRequest("vpnn", "calc_losses", pn.nnconfig);
			}
			
			
			
	}
		
		
		if (CDState.getCycle() == cycles-1 && pn.getID() == Network.size()-1 ) {
			// plot loss curve
			System.out.println("Cycles " + cycles);
			HTTPSendDetailsAtOnce.sendRequest("vpnn", "plot", pn.nnconfig);
		}

	}

	/**
	 * Selects a random neighbor from those stored in the {@link Linkable} protocol
	 * used by this protocol.
	 */
	protected Node selectNeighbor(Node node, int pid) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return linkable.getNeighbor(
					CommonState.r.nextInt(linkable.degree()));
		else
			return null;
	}

	public static void writeIntoFile(String millis) {
		File file = new File("exec-time.txt");
		 
		// if file doesnt exists, then create it
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		FileWriter fw;
		try {
			fw = new FileWriter(file.getAbsoluteFile(),true);

		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(millis+"\n");
		bw.close();
		} catch (IOException e)
		
		 {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		

	}
	
	private INDArray crossEntropyLoss(INDArray predictions, INDArray labels) {

		/*
		 Computes the cross-entropy loss between predictions and labels array.
		 */
		int numRows = predictions.rows();
		int numCols = predictions.columns();
		
		INDArray batchLossVector = Nd4j.zeros(numRows, 1);
		for(int i=0;i<numRows;i++) {
			double loss = 0.0;
			for(int j=0;j<numCols;j++) {
				loss += ((labels.getDouble(i,j)) * Math.log(predictions.getDouble(i,j) + 1e-15)) + (
						(1-labels.getDouble(i,j)) * Math.log((1-predictions.getDouble(i,j)) + 1e-15)
						)
						;
				
			}
			batchLossVector.putScalar(i, 0, -loss);
		}
		return batchLossVector;
	}
}
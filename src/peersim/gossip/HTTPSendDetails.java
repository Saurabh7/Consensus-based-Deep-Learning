package peersim.gossip;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.Socket;
import java.net.URL;
import java.net.URLEncoder;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

public class HTTPSendDetails {
	HttpURLConnection connection = null;
	DataOutputStream wr = null;
	BufferedReader rd = null;
	 InputStream is = null;
	 StringBuffer response = null;
	public void openConnection() {
		
		
		try {
			  
	          //Create connection
			  URL url = new URL("http://127.0.0.1:5000/update_project/"+"updateNN");
			  String urlParameters = "command=" + "hello"+"+"+"node_id="+1;
	          connection = (HttpURLConnection)url.openConnection();
	          connection.setRequestMethod("POST");
	          connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

	          connection.setRequestProperty("Content-Length", "" + Integer.toString(urlParameters.getBytes().length));
	          connection.setRequestProperty("Content-Language", "en-US");  

	          connection.setUseCaches(false);
	          connection.setDoInput(true);
	          connection.setDoOutput(true);
	          
	          wr = new DataOutputStream (connection.getOutputStream ());
	          	//Get Response    	
			   is = connection.getInputStream();
			    rd = new BufferedReader(new InputStreamReader(is));
			    
			    response = new StringBuffer();
	          
	        } catch (Exception e) {
	            e.printStackTrace();
	            //return null;
	        }finally {
	            
	        }
	}
	
	
	public void sendCommand(String command, int node_id) {
        /*
         * This function is used to send commands to the flask-Pytorch ML interface.
         */
		
		String urlParameters = "command=" + command+"+"+"node_id="+node_id;
		
      //Send request
		
		try {
		    System.out.println("Writing: " + urlParameters);
		    wr.writeBytes (urlParameters);
//		    wr.flush ();

		
		    
		    String line;
		     
		    while((line = rd.readLine()) != null) {
		     System.out.println("Got response: "+ line);
		      response.append(line);
		      response.append('\r');
		    }

		}
		catch (Exception e) {
            e.printStackTrace();
            //return null;
		}
        
    }
	
	void closeConnection() {
		try {
			wr.close();
		    rd.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String args[]) {
		HTTPSendDetails newConnection = new HTTPSendDetails();
		newConnection.openConnection();
		newConnection.sendCommand("hello", 1);
		newConnection.sendCommand("hello", 2);
		newConnection.sendCommand("hello", 3);
		newConnection.closeConnection();
	}
}

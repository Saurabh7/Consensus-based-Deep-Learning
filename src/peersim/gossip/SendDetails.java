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

public class SendDetails {
	Socket socket = null;
	BufferedWriter wr = null;
	BufferedReader rd = null;
	public void openConnection() {
		
		
		try {
			  
	          //Create connection
	          String host = "127.0.0.1";
//	          
//	          connection = (HttpURLConnection)url.openConnection();
//	          connection.setRequestMethod("POST");
//	          connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
//
//	          
//	          connection.setRequestProperty("Content-Language", "en-US");  
//
//	          connection.setUseCaches(false);
//	          connection.setDoInput(true);
//	          connection.setDoOutput(true);
	          socket = new Socket(host, 5000);
	          wr = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), "UTF8"));
	          rd = new BufferedReader(new InputStreamReader(socket.getInputStream()));
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
		
		String data = "command=" + command+"+"+"node_id="+node_id;
//		connection.setRequestProperty("Content-Length", "" + Integer.toString(urlParameters.getBytes().length));
      //Send request
		
//		try {
//		    DataOutputStream wr = new DataOutputStream (connection.getOutputStream ());
//		    wr.writeBytes (urlParameters);
//		    wr.flush ();
//		    wr.close ();
//		
//		    //Get Response    
//		    InputStream is = connection.getInputStream();
//		    BufferedReader rd = new BufferedReader(new InputStreamReader(is));
//		    String line;
//		    StringBuffer response = new StringBuffer(); 
//		    while((line = rd.readLine()) != null) {
//		      response.append(line);
//		      response.append('\r');
//		    }
//		    rd.close();
//		}
//		catch (Exception e) {
//            e.printStackTrace();
//            //return null;
//		}
        try {
			String path = "/update_project/updateNN";
		    
		    wr.write("POST " + path + " HTTP/1.0\r\n");
		    wr.write("Content-Length: " + data.length() + "\r\n");
		    wr.write("Content-Type: application/x-www-form-urlencoded\r\n");
		    wr.write("\r\n");
		
		    wr.write(data);
		    wr.flush();
		
		    
	    String line;
	    while ((line = rd.readLine()) != null) {
	      System.out.println(line);
	    }
	    
	    
        }
        catch(Exception e) {
          e.printStackTrace();
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
	
}

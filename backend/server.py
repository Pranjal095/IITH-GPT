import json
import sys
from flask_cors import CORS  # Ensures proper handling of CORS

from flask import Flask, request, jsonify
from IITH_GPT.Agentic_code.RAG_LLM import process_query_with_validation

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        print(data)
        # Check if the query is present
        if not data or 'query' not in data:
            return jsonify({"error": "Invalid request: 'query' field is required"}), 400

        user_query = data['query']
        
        # Get the bot response using the function from IITH_GPT
        bot_response = process_query_with_validation(query=user_query)
        print(bot_response)  # Log the response (useful for debugging)

        # Return the response to the client
        return jsonify({"response": bot_response})
    
    except Exception as e:
        # Error handling: log the exception and return an appropriate error message
        if app.debug:  # Only show detailed error in debug mode
            print(f"Error occurred: {e}")  # Log the error to the console for debugging
            return jsonify({"error": f"Something went wrong: {str(e)}"}), 500
        
        # Return a generic error message for production
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    # Running the application on all available interfaces (host='0.0.0.0') and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)

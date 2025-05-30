from flask import Flask, request, jsonify
from flask_cors import CORS
import classification
import os

app = Flask(__name__)
CORS(app)  

@app.route('/api/classify', methods=['POST'])
def classify_email():
    data = request.json
    
    if not data or ('subject' not in data and 'body' not in data):
        return jsonify({'error': 'Missing email content'}), 400
    
    subject = data.get('subject', '')
    body = data.get('body', '')
    
    email = f"""
    Subject: {subject}
    Body: {body}
    """
    
    result = classification.classify(email)
    
    return result

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

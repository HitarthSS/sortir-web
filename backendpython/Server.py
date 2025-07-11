from flask import Flask, request, jsonify
from VideoHelper import process_video

app = Flask(__name__)

@app.route('/process_video', methods=['POST'])
def process_video_route():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    result = process_video(url)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 
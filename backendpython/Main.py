import requests

url = 'http://127.0.0.1:5001/process_video'
video_url = input('Enter video URL: ')
data = {'url': video_url}

response = requests.post(url, json=data)

print('Response from server:')
print(response.json()) 
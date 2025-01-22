from pymongo import MongoClient
import time
import pandas as pd
import io
import base64
import json
import requests

def fetch_data_from_mongodb(collection_name):
    # Start of Selection
    # 连接到MongoDB
    client = MongoClient('10.215.4.34:27017')
    db = client['tool']
    
    # 获取数据
    collection = db[collection_name]  # 使用传入的集合名称
    data = list(collection.find({}, {'_id': False}))
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data)
    
    # 根据时间戳生成唯一的文件名
    timestamp = int(time.time())
    filename = f'output_{timestamp}.csv'
    
    # 将DataFrame保存为CSV格式
    df.to_csv(filename, index=False)
    return filename

# 上传
def upload_file(base64_image):
    url = "http://10.215.4.34:8080/file/commonFile/upload"
    img_data = io.BytesIO(base64.b64decode(base64_image))
    files = {'file': ('image.png', img_data, 'image/png')}
    response = requests.post(url, files=files)
    response_json = json.loads(response.text)
    print('response_json:', response_json)
    file_path = ''
    if 'data' in response_json and response_json['data'] is not None:
        data = response_json['data']
        if 'filePath' in data:
            file_path = data['filePath']
    return file_path


from flask import Flask, request, jsonify
from knn import knn_analysis

app = Flask(__name__)

@app.route('/knn_analysis', methods=['POST'])
def knn_analysis_endpoint():
    if 'collect_name' not in request.json:
        return jsonify({'code': 500, 'msg': 'collect_name is required'})
    
    collect_name = request.json['collect_name']
    try:
        file_path = fetch_data_from_mongodb(collect_name)
        result_json = knn_analysis(file_path)
        result = json.loads(result_json)  # 将JSON字符串解析为Python对象
        # 上传每个图像并重新封装数据
        all_data = []
        for item in result:
            img_path = upload_file(item['base64'])
            all_data.append({"path": img_path, "type": "img", "title": item['title']})
        
        print('all_data:', all_data)
        
        # 返回JSON响应
        return jsonify({"code": 200, "msg": "success", "data": {"list": all_data}})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"处理过程中发生错误: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

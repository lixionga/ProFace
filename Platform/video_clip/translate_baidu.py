import requests
import json
API_KEY = "5yTCrRyKKcmW3GGAxrQtjnAS"
SECRET_KEY = "zoG1REY3Eals2DbERTvd4WKhASMP601v"

def zh2en(zh_str):
    url = "https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=24.e72d75078e8d0d6f86528bc3e93340fd.2592000.1691826644.282335-26894970"

    payload = json.dumps({
        "from": "zh",
        "to": "en",
        "q": zh_str
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)

    en_str = result["result"]["trans_result"][0]["dst"]
    return en_str



# Access Token获取
# def get_access_token():
#     """
#     使用 AK，SK 生成鉴权签名（Access Token）
#     :return: access_token，或是None(如果错误)
#     """
#     url = "https://aip.baidubce.com/oauth/2.0/token"
#     params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
#     return str(requests.post(url, params=params).json().get("access_token"))




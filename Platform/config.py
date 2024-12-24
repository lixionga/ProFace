class Config:
    SECRET_KEY = 'your-secret-key'  # 请更改为安全的密钥
    DEBUG = True
    
    # Flask配置
    FLASK_ENV = 'development'
    JSON_AS_ASCII = False
    
    # 模型配置
    MODEL_PATH = 'model/nlp_structbert_emotion-classification_chinese-base'
    
    # 其他配置项...

config = Config()  # 创建Config实例

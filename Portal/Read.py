import json
import torch
def load_embs_features():
    with open('/home/chenyidou/x_test/web/id_embs.json', 'r') as f:
        stored_embs = json.load(f)

    with open('/home/chenyidou/x_test/web/id_features.json', 'r') as f:
        stored_feats = json.load(f)

    embs_tensor = torch.tensor(stored_embs)
    feats_tensor = torch.tensor(stored_feats)
    return embs_tensor, feats_tensor

# 使用示例
if __name__ == '__main__':
    embs, feats = load_embs_features()
    print(feats[0].shape)
    print("读取到的embs数量:", len(embs))
    print("读取到的feats数量:", len(feats))
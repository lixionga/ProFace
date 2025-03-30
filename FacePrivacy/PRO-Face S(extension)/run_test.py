from utils.image_processing import *
from PIL import Image
from torchvision import transforms

if __name__ == "__main__":
    image_path = '...'
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to('cuda:0')
    z, dis_target, \
        rand_z, rand_dis_target, \
        inv_z, inv_dis_target, \
        rand_inv_z, rand_inv_dis_target, \
        rand_inv_2nd_z, _ = generate_code(passwd_length=16,
                                          batch_size=8,
                                          device='cuda:0',
                                          inv=True,
                                          use_minus_one='no',
                                          gen_random_WR=True)



    model = PreProcessing(device="cuda:0")
    anoy = model(image_tensor, z)
    print("down!")

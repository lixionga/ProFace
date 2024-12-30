import os
from argparse import Namespace
import sys

sys.path.append("/data/lk/ID-disen4")


# sys.path.append("..")
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm
from configs.test_options import TestOptions
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio,MeanAbsoluteError
from utils.utils_metric import MyMetric
from models.id_encoder import id_loss
# print(sys.path)
from models.attr_encoder.psp import pSp
from models.id_transformer.inn import bulid_models
from models.mlp.LatentMapper import LatentMapper, FusionMapper10
from models.attr_encoder.StyleGan2.model import Generator
from models.deghosting.deghosting import Deghosting
from utils.utils import loading_pretrianed, get_concat_vec, normalize, setup_seed
from utils.utils_data import ImageWithMaskDataset, get_masked_imgs, ImageAndMask_Dataset
from torchvision.utils import save_image

#加载模型出错， test_options参数有问题


def run(opts, IMAGE_DATA_DIR, MASK_DIR, W_DATA_DIR):
    # opts = Namespace(**opts)
    device = opts.device

    out_path_results = os.path.join(os.path.dirname(os.path.dirname(__file__)), opts.exp_dir)#, opts.editing_type, opts.input_type
    print(f'out_path_results = {out_path_results}')
    os.makedirs(out_path_results, exist_ok=True)

    #loading models
    id_encoder = id_loss.IDLoss(opts.id_encoder_path).to(device).eval()
    attr_encoder = pSp(opts).to(device).eval()
    id_transfomer = bulid_models().to(device).eval()
    mlp = LatentMapper().to(device).eval()
    generator = Generator(opts.image_size, 512, 8).to(device).eval()
    icl = FusionMapper10().to(device).eval()
    # attr_encoder = torch.load(opts.attr_encoder_path)
    # mlp = torch.load(opts.mlp_path)
    # attr_encoder = torch.load('/data/lk/ID-disen4/pretrained_models/ckptattr_encoder_MKIHUPYWYPCR_2301.pt')
    # mlp = torch.load('/data/lk/ID-disen4/pretrained_models/ckptmaper_MKIHUPYWYPCR_23010.pt',)
    mlp_dict = torch.load('/data/lk/ID-disen4/pretrained_models/mlp_best.pth')
    attr_dict = torch.load('/data/lk/ID-disen4/pretrained_models/attr_best.pth')
    G_pretrained_dict = torch.load(opts.id_transformer_path)# train 15 id cos 0.49400532 ssim 0.94676554
    fuse_pretrained_dict = torch.load(opts.icl_path)
    state_dict = torch.load(opts.stylegan_weights)
    loading_pretrianed(attr_dict, attr_encoder)
    loading_pretrianed(mlp_dict, mlp)
    loading_pretrianed(G_pretrained_dict, id_transfomer)
    loading_pretrianed(fuse_pretrained_dict, icl)					
    generator.load_state_dict(state_dict['g_ema'], strict=False)
    degh_net = Deghosting(in_size=128, out_size=256, pretrain=opts.deghosting_path).to(device).eval()

    #loading metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse = MeanSquaredError().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    mae = MeanAbsoluteError().to(device)
    metric_ssim = MyMetric(ssim)
    metric_mse = MyMetric(mse)
    metric_psnr = MyMetric(psnr)
    metric_mae = MyMetric(mae)

    # landmark_encoder = Landmark_Encoder.Encoder_Landmarks(opts.landmark_encoder_path).to(device).eval()
    # loss_fn_vgg = lpips.LPIPS(net='alex').to(device).eval()
    # loss_mse = torch.nn.MSELoss().to(device).eval().to(device).eval()

    #setting dataloader
    if opts.data_name == 'Recovery&Disentangled_FFHQ*':
        w_image_dataset = ImageWithMaskDataset(W_DATA_DIR, IMAGE_DATA_DIR, MASK_DIR)
        # train_size = int(0.86* len(w_image_dataset))
        # test_size = len(w_image_dataset) - train_size
        # train_data, test_data = random_split(w_image_dataset, [train_size, test_size])
        batch_data_loader = DataLoader(dataset=w_image_dataset, batch_size=opts.batch_size, shuffle=False,drop_last=True)
    else:
        w_image_dataset = ImageAndMask_Dataset(IMAGE_DATA_DIR, MASK_DIR)
        batch_data_loader = DataLoader(dataset=w_image_dataset, batch_size=opts.batch_size, shuffle=False,drop_last=False)


    for i_batch, data_batch in tqdm(enumerate(batch_data_loader)):
        if opts.data_name == 'Recovery&Disentangled_FFHQ*':
            _, images, masks = data_batch
        else:
            images, masks = data_batch
        if opts.use_one_batch == True:
            if i_batch > 19 :
                break

        test_id_images = images.to(device)
        x_ori = test_id_images
        test_masks = masks.to(device)

        use_512bitpasswords = True
        # set up passwords
        if use_512bitpasswords == True:

            passwords_fir = (torch.rand((1,512),device=device)+1.)/2.
            passwords_fir, _ = torch.broadcast_tensors(passwords_fir, torch.rand((opts.batch_size,512),device=device))
            passwords_fir = generator.style(passwords_fir)

            passwords_thr = (torch.rand((1,512),device=device)+1.)/2.   
            passwords_thr, _ = torch.broadcast_tensors(passwords_thr, torch.rand((opts.batch_size,512),device=device))
            passwords_thr = generator.style(passwords_thr)

            while 1:
                passwords_sec = (torch.rand((1,512),device=device)+1.)/2.   
                passwords_sec, _ = torch.broadcast_tensors(passwords_sec, torch.rand((opts.batch_size,512),device=device))
                passwords_sec = generator.style(passwords_sec)
                if passwords_fir[0][0] != passwords_sec[0][0] or passwords_fir[0][1] != passwords_sec[0][1]:
                    break            

        else:

            ValueError()
        # config
        with_passwords = 1
        ############distangled############
        if with_passwords == True:
            
        # #########################test  sne########################
        #         if use_SNE == True:
        #             with torch.no_grad():
        #                 id_vec = id_encoder.extract_feats((test_id_images*2.)-1.)
        #                 for sne_idx in range(100):

        #                     ps_sne = (torch.rand((1,512),device=Global_Config.device)+1.)/2.
        #                     ps_sne, _ = torch.broadcast_tensors(ps_sne, torch.rand((config['batchSize'],512),device=Global_Config.device))
        #                     ps_sne = generator.style(ps_sne)
                            
        #                     id_fake, _ = G_net(id_vec, c=[ps_sne])
        #                     if sne_idx == 0:
        #                         ids = id_fake
        #                     else :
        #                     #     print(f'id_fake shape = {id_fake.shape},ids shape = {ids.shape} ')
        #                     #     a
        #                         ids = torch.cat((ids,id_fake),dim=0)
        #                 ids = ids.cpu().numpy()
        #                 print(f'id shape = {ids.shape}')
        #                 np.save(f'id_{i_batch}', ids)

        # #########################test  sne########################


    # #########################kffa########################
            # with torch.no_grad():
            #     # id_vec = id_encoder.extract_feats((test_id_images*2.)-1.)
            #     # cos_list = []
            #     for sne_idx in range(50):
            #         ps_sne = None
            #         ps_sne = (torch.rand((1,512),device=Global_Config.device)+1.)/2.
            #         ps_sne, _ = torch.broadcast_tensors(ps_sne, torch.rand((config['batchSize'],512),device=Global_Config.device))
            #         ps_sne = generator.style(ps_sne)
                
            #         kffa_w, _, _, _ = get_concat_vec(test_id_images, test_id_images, id_encoder, attr_encoder, ps_sne, mode='forward')
            #         # kffa_id = torch.mean(kffa_id, dim=1)
            #         with torch.no_grad():
            #             kffa_code = mlp(kffa_w)
            #             kffa_imgs_tensor, _ = generator([kffa_code], input_is_latent=True, return_latents=False, randomize_noise=False)

            #         kffa_imgs = get_masked_imgs((kffa_imgs_tensor+1)/2, test_id_images, test_masks)

            
            #         save_image(kffa_imgs, os.path.join(f'/home/yl/lk/code/ID-disen_shows/ablation_shows/{Alation_Path}/KFFA/imgs/{str(i_batch+1)}_Ours_{str(sne_idx+1)}.jpg'))
            #         # save_image(kffa_imgs, os.path.join(f'/home/yl/lk/code/ID-disen_shows/anonymized_shows/Ours/{str(i_batch+1)}_Ours_{str(sne_idx+1)}.jpg'))


                #     concat_rand_recon_vec_cycled, rand_recon_id, rand_recon_attr, rand_recon_id_before = get_concat_vec(kffa_imgs_tensor,kffa_imgs_tensor,id_encoder, attr_encoder,passwords_sec,mode='backward')
                #     with torch.no_grad():
                #         mapped_concat_rand_recon_vec_cycled = mlp(concat_rand_recon_vec_cycled)   
                #         other_recon_imgs, _ = generator([mapped_concat_rand_recon_vec_cycled], input_is_latent=True, return_latents=True, randomize_noise=False)
                #         rand_recon = get_masked_imgs((other_recon_imgs+1)/2, test_id_images, test_masks)
                #         save_image(kffa_imgs, os.path.join(f'/home/yl/lk/code/ID-disen_shows/anonymized_shows/Ours/{str(i_batch+1)}_Ours_wrong_recon_{str(sne_idx+1)}.jpg'))
                #     kffa_id = id_encoder.extract_feats(kffa_imgs)
                #     if sne_idx == 0:
                #         ids = kffa_id
                #     else :
                #         ids = torch.cat((ids,kffa_id),dim=0)
                #     # ids = ids.cpu().numpy()
                #     # np.save(f'id_{i_batch}', ids)
                # if i_batch == 0:
                #     total_ids = ids
                # else :
                #     total_ids = torch.cat((total_ids,ids),dim=0)
                    
                # print(f'total_ids shape = {total_ids.shape}')

                
                    # if sne_idx >0:
                    #     cos_similarity = torch.cosine_similarity(ids[sne_idx-1].unsqueeze(0), ids[sne_idx].unsqueeze(0), dim=1)
                    #     cos_list.append(cos_similarity)
                    #     print(f'cos_similarity = {cos_similarity}')
                
                # print(f'cos_list.mean() = {cos_list.mean()}')
    # #########################kffa########################       

            # #inversion
            # concat_inv = inversion_encoder(test_id_images, test_id_images, id_encoder, attr_encoder)
            # with torch.no_grad():
            #     mapped_concat_vec_inv = mlp(concat_inv)
            #     inv_imgs, _ = generator([mapped_concat_vec_inv], input_is_latent=True, return_latents=False, randomize_noise=False)
            #     # inve = (inv_imgs+1.)/2
            #     inve = get_masked_imgs((inv_imgs+1)/2, test_id_images, test_masks)

            with torch.no_grad():
            #first generated
                concat_vec_cycled_fir, fake_id_fir, orin_attr, fake_id_before = get_concat_vec(test_id_images, test_id_images, id_encoder, attr_encoder, id_transfomer, icl, passwords_fir, mode='forward')
                mapped_concat_vec_fir = mlp(concat_vec_cycled_fir)
                anonymous_fir_imgs, _ = generator([mapped_concat_vec_fir], input_is_latent=True, return_latents=False, randomize_noise=False)
                generated_images_tensor = get_masked_imgs((anonymous_fir_imgs+1)/2, test_id_images, test_masks)
            
            #random generated
                concat_vec_cycled_sec, fake_id_sec, _, _ = get_concat_vec(test_id_images, test_id_images, id_encoder, attr_encoder, id_transfomer, icl, passwords_sec, mode='forward')
                mapped_concat_vec_sec = mlp(concat_vec_cycled_sec)
                anonymous_sec_imgs, _ = generator([mapped_concat_vec_sec], input_is_latent=True, return_latents=False, randomize_noise=False)
                generated_sec = get_masked_imgs((anonymous_sec_imgs+1)/2, test_id_images, test_masks)

            # recon
                concat_recon_vec_cycled, recon_id, recon_attr, recon_id_before = get_concat_vec(generated_images_tensor, generated_images_tensor, id_encoder, attr_encoder, id_transfomer, icl, passwords_fir, mode='backward')
                mapped_concat_recon_vec_cycled = mlp(concat_recon_vec_cycled)
                recon_imgs, _ = generator([mapped_concat_recon_vec_cycled], input_is_latent=True, return_latents=True, randomize_noise=False)
                recon = get_masked_imgs((recon_imgs+1)/2, test_id_images, test_masks)

            # 
            # # rand recon
                concat_rand_recon_vec_cycled, rand_recon_id, rand_recon_attr, rand_recon_id_before = get_concat_vec(generated_images_tensor,generated_images_tensor,id_encoder, attr_encoder, id_transfomer, icl,passwords_sec,mode='backward')
                mapped_concat_rand_recon_vec_cycled = mlp(concat_rand_recon_vec_cycled)   
                other_recon_imgs, _ = generator([mapped_concat_rand_recon_vec_cycled], input_is_latent=True, return_latents=True, randomize_noise=False)
                rand_recon = get_masked_imgs((other_recon_imgs+1)/2, test_id_images, test_masks)

            #de_attr
                # concat_de_attr = de_attr_encoder(test_id_images, test_id_images, id_encoder, attr_encoder)
                # mapped_concat_vec_de_attr = mlp(concat_de_attr)
                # de_attr, _ = generator([mapped_concat_vec_de_attr], input_is_latent=True, return_latents=False, randomize_noise=False)


            #de_id
                # concat_de_id = de_id_encoder(test_id_images, test_id_images, id_encoder, attr_encoder)
                # mapped_concat_vec_de_id = mlp(concat_de_id)
                # de_id, _ = generator([mapped_concat_vec_de_id], input_is_latent=True, return_latents=False, randomize_noise=False)


            save_image(torch.cat((x_ori,generated_images_tensor,generated_sec,recon, rand_recon), dim=0), os.path.join(f'{out_path_results}/test_FFHQ_Metric{i_batch}.jpg'), nrow=int(opts.batch_size))

            # SAVE_IMG_PATH ="/home/yl/lk/code/ID-disen_shows/ablation_shows/ONE_STAGE_WO"
            # #save img 
            # save_image(x_ori, os.path.join(f'{SAVE_IMG_PATH}/TEST_DATA/orig/{str(i_batch+1)}_Orig.jpg'))
            # save_image(inve, os.path.join(f'{SAVE_IMG_PATH}/inversion/{str(i_batch+1)}_Ours_1.jpg'))
            # save_image(generated_images_tensor, os.path.join(f'{SAVE_IMG_PATH}/Anon1/{str(i_batch+1)}_Ours_1.jpg'))
            # save_image(generated_sec, os.path.join(f'{SAVE_IMG_PATH}/Anon2/{str(i_batch+1)}_Ours_1.jpg'))
            # save_image(recon, os.path.join(f'{SAVE_IMG_PATH}/Recon/{str(i_batch+1)}_Ours_1.jpg'))
            # save_image(rand_recon, os.path.join(f'{SAVE_IMG_PATH}/RandRecon/{str(i_batch+1)}_Ours_1.jpg'))
            # save_image(test_masks, os.path.join(f'{SAVE_IMG_PATH}/TEST_DATA/mask/ {str(i_batch+1)}_Orig.jpg'))

            # save_image((de_attr+1.)/2., os.path.join(f'/home/yl/lk/code/ID-disen_shows/Disentangled_shows/Ours/De_attr/{str(i_batch+1)}_Ours_1.jpg'))
            # save_image((de_id+1.)/2., os.path.join(f'/home/yl/lk/code/ID-disen_shows/Disentangled_shows/Ours/De_id/{str(i_batch+1)}_Ours_1.jpg'))
            # save_image(test_masks, os.path.join(f'/home/yl/lk/code/ID-disen_shows/Recovery_shows/TEST_DATA/mask/ {str(i_batch+1)}_Orig.jpg'))
            #save 
            # total_grad = torch.concat((x_ori, generated_images_tensor, recon), dim=0)
            # save_image(total_grad, f'/home/yl/lk/code/ID-dise4/metrics/{str(i_batch)}_total.jpg',nrow=4)
            

            # save_image(x_ori,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Orin.png')
            # save_image(generated_images_tensor,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Anon.png')
            # save_image((anonymous_fir_imgs+1)/2,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_nomask_Anon.png')
            # save_image(generated_sec,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_RandAnon.png')
            # save_image((recon_imgs+1)/2,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_nomask_Recon.png')
            # save_image(recon,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Recon.png')
            # save_image(rand_recon,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_RandRecon.png')
            # save_image(inve,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Inversion.png')

            # save_image(test_masks,f'/home/yl/lk/code/ID-dise4/visual/IDSF-FFHQ/FFHQ_{str(i_batch)}_Mask.png')


    # ####################################################### 计算指标
            # Recovery quality x_ori
            metric_ssim._method(normalize(recon.to(device)), normalize(x_ori.to(device)))
            # metric_mse._method(normalize(recon.to(device)), normalize(x_ori.to(device)))
            metric_psnr._method(normalize(recon.to(device)), normalize(x_ori.to(device)))
            metric_mae._method(normalize(recon.to(device)), normalize(x_ori.to(device)))
    print(f'metric_ssim:{metric_ssim._target_out()},metric_psnr:{metric_psnr._target_out()},metric_mae:{metric_mae._target_out()}')
            # arc_fake_embedding = id_encoder.extract_feats(recon*2-1)# imgs x_ori
            # arc_orig_embedding = id_encoder.extract_feats(x_ori*2-1)# fake_save*2-1 fake
            # recon_arc_cos_similarity = torch.cosine_similarity(arc_fake_embedding, arc_orig_embedding, dim=1)
            # recon_arc_cos_similarity_list.append(recon_arc_cos_similarity.abs().detach().cpu().numpy())  

            # Fake quality fake

            # fake_ssim_score = ssim(normalize(generated_images_tensor.to(device)), normalize(x_ori.to(device))).cpu()
            # fake_ssim_list.append(fake_ssim_score)

            # fake_mse = mse(normalize(generated_images_tensor.to(device)), normalize(x_ori.to(device))).cpu()
            # fake_mse_list.append(fake_mse)

            # fake_psnr = psnr(generated_images_tensor.to(device), x_ori.to(device)).cpu()
            # fake_psnr_list.append(fake_psnr)

            # fake_lpips = lpips_loss(normalize(generated_images_tensor.to(device)), normalize(x_ori.to(device))).detach().cpu()
            # fake_lpips_list.append(fake_lpips)
            

            # from facenet_pytorch import MTCNN, InceptionResnetV1
            # IMAGE_SIZE = 220
            # mtcnn = MTCNN(
            #     image_size=IMAGE_SIZE, margin=0, min_face_size=20,
            #     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            #     device=Global_Config.device
            # )
            # to_pil = transforms.ToPILImage(mode='RGB')
            #mtcnn ori_bboxes[0][0] 代表检测框左边，ori_bboxes[0][1]代表人脸被检测到的概率， ori_bboxes[0][2]代表人脸关键点
            # ori_bboxes = [mtcnn.detect(to_pil(image),landmarks=True) for image in (x_ori+1)/2]
            # fake_bboxes = [mtcnn.detect(to_pil(image),landmarks=True) for image in generated_images_tensor] #generated_images_tensor (anonymous_fir_imgs+1)/2
            # try:
            #     Face_detection = (calculate_iou(ori_bboxes[0][0], fake_bboxes[0][0]))
            #     Face_detection_list.append(Face_detection)
            # except Exception as e:
            #     print(f"An error occurred: {e}")

            # Bounding_box_distance = (abs(fake_bboxes[0][1]-ori_bboxes[0][1]))
            # Bounding_box_distance_list.append(Bounding_box_distance)

            # Landmark_distance = (abs(ori_bboxes[0][2] - fake_bboxes[0][2]))
            # Landmark_distance_list.append(Landmark_distance)
            
            # #arcface 
            # arc_fake_embedding = id_encoder.extract_feats(anonymous_fir_imgs)#generated_images_tensor*2-1 anonymous_fir_imgs
            # arc_orig_embedding = id_encoder.extract_feats(x_ori*2-1)
            # arc_cos_similarity = torch.cosine_similarity(arc_fake_embedding, arc_orig_embedding, dim=1) #(fake_id_fir.mean(1), fake_id_before.mean(1), dim=1)
            # arc_cos_similarity_list.append(arc_cos_similarity.detach().cpu().numpy())
                
            # ada_model = load_pretrained_model('ir_50').to(Global_Config.device)
            # def fix_align(input_imgs):
            #     input_imgs = input_imgs[:, :, 35:223, 32:220]
            #     with torch.no_grad():
            #         face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
            #         out_imgs = face_pool(input_imgs)
            #     return out_imgs
            # ada_orig_feature, _ = ada_model(fix_align(generated_images_tensor*2-1)) #generated_images_tensor*2-1 anonymous_fir_imgs
            # ada_fake_feature, _ = ada_model(fix_align(x_ori*2-1))
            # ada_cos_similarity = torch.cosine_similarity(ada_orig_feature, ada_fake_feature, dim=1) #(fake_id_fir.mean(1), fake_id_before.mean(1), dim=1)
            # ada_cos_similarity_list.append(ada_cos_similarity.detach().cpu().numpy())       

            # ada_anon_mae = mae(normalize(ada_orig_feature.to(device)), normalize(ada_fake_feature.to(device))).cpu().detach().numpy()
            # ada_anon_mae_list.append(ada_anon_mae)   


            # # #calc SIT
            # #z,z..
            # Zfake_cos_similarity = torch.cosine_similarity(fake_id_fir.mean(1), fake_id_before.mean(1), dim=1) 
            # Zfake_cos_similarity_list.append(Zfake_cos_similarity.detach().cpu().numpy())

            # #z,zf
            # Zrecon_cos_similarity = torch.cosine_similarity(recon_id.mean(1), fake_id_before.mean(1), dim=1) 
            # Zrecon_cos_similarity_list.append(Zrecon_cos_similarity.detach().cpu().numpy())

            # #z..,zf
            # fake2_cos_similarity = torch.cosine_similarity(recon_id.mean(1), recon_id_before.mean(1), dim=1) 
            # fake2_cos_similarity_list.append(fake2_cos_similarity.detach().cpu().numpy())

            # #z..,z..
            # fakeZ_cos_similarity = torch.cosine_similarity(fake_id_fir.mean(1), recon_id_before.mean(1), dim=1) 
            # fakeZ_cos_similarity_list.append(fakeZ_cos_similarity.detach().cpu().numpy())

            #calc norm feature
            # arc_anon_mse = mse(arc_fake_embedding.to(device), arc_orig_embedding.to(device)).cpu().detach().numpy()
            # arc_anon_mse_list.append(arc_anon_mse)

            # arc_anon_mae = mae(arc_fake_embedding.to(device), arc_orig_embedding.to(device)).cpu().detach().numpy()
            # arc_anon_mae_list.append(arc_anon_mae)     


            # #vggface2
            # vgg_face_net = InceptionResnetV1(pretrained='vggface2').eval().to(Global_Config.device)
            # vgg_fake_cropped = mtcnn(to_pil(generated_images_tensor.reshape(3,256,256)))
            # vgg_fake_embedding = vgg_face_net(vgg_fake_cropped.unsqueeze(0).to(Global_Config.device))
            # vgg_orig_cropped = mtcnn(to_pil(test_id_images.reshape(3,256,256)))
            # vgg_orig_embedding = vgg_face_net(vgg_orig_cropped.unsqueeze(0).to(Global_Config.device))

            # vgg_cost_similarity = torch.cosine_similarity(vgg_fake_embedding, vgg_orig_embedding, dim=1)
            # vgg_cos_similarity_list.append(vgg_cost_similarity.abs().detach().cpu().numpy())


            # # vgg_anon_mse = mse(normalize(vgg_fake_embedding.to(device)), normalize(vgg_orig_embedding.to(device))).cpu().detach().numpy()
            # # vgg_anon_mse_list.append(vgg_anon_mse)
            
            # vgg_anon_mae = mae(normalize(vgg_fake_embedding.to(device)), normalize(vgg_orig_embedding.to(device))).cpu().detach().numpy()
            # vgg_anon_mae_list.append(vgg_anon_mae)                   


    #         #casia
    #         casia_face_net = InceptionResnetV1(pretrained='casia-webface').eval().to(Global_Config.device)
    #         casia_fake_cropped = mtcnn(to_pil((anonymous_fir_imgs.reshape(3,256,256)+1)/2.))
    #         casia_fake_embedding = casia_face_net(casia_fake_cropped.unsqueeze(0).to(Global_Config.device))
    #         casia_orig_cropped = mtcnn(to_pil(test_id_images.reshape(3,256,256)))
    #         casia_orig_embedding = casia_face_net(casia_orig_cropped.unsqueeze(0).to(Global_Config.device))
    # #         # print(f'aa')
    #         casia_cost_similarity = torch.cosine_similarity(casia_fake_embedding, casia_orig_embedding, dim=1)
    #         casia_cos_similarity_list.append(casia_cost_similarity.detach().cpu().numpy())

    #         anon_mse = mse(normalize(generated_images_tensor.to(device)), normalize(test_id_images.to(device))).cpu().detach().numpy()
    #         anon_mse_list.append(anon_mse)
            
    #         anon_mae = mae(normalize(generated_images_tensor.to(device)), normalize(test_id_images.to(device))).cpu().detach().numpy()
    #         anon_mae_list.append(anon_mae)


    # ##########################################################

            # save_image((imgs + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_orin.jpg')
            # save_image((fake + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_fake.jpg')
            # save_image((recon + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_recon.jpg')
            # save_image((wrong_recon + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_wrong_recon.jpg')


    #         # save_image((test_imgs + 1) / 2, f'temp_images/test_{i_batch}.jpg', nrow=4)
    #         # save_image((x_ori + 1) / 2, f'temp_images/x_ori_{i_batch}.jpg', nrow=4)
    #         # save_image((imgs + 1) / 2, f'temp_images/orig_{i_batch}.jpg', nrow=4)
    #         # save_image((fake + 1) / 2, f'temp_images/fake_{i_batch}.jpg', nrow=4)
    #         # save_image((rand_fake + 1) / 2, f'temp_images/rand_fake_{i_batch}.jpg', nrow=4)
    #         # save_image((recon + 1) / 2, f'temp_images/recon_{i_batch}.jpg', nrow=4)
    #         # save_image((wrong_recon + 1) / 2, f'temp_images/wrong_recon_{i_batch}.jpg', nrow=4)

    # total_ids = total_ids.cpu().numpy()
    # np.save(f'/home/yl/lk/code/ID-dise4/Training/TSNE_OUT/idsf/TSNE_feature_data.npy', total_ids)
    # np.save(f'/home/yl/lk/code/ID-disen_shows/ablation_shows/DIV_WO/KFFA/data/TSNE_feature_data.npy', total_ids)
    # metric_results = {
    #     # 'privacy_metric': np.mean(privacy_metric_list),
    #     'recovery_ssim': np.mean(rec_ssim_list),
    #     'recovery_lpips': np.mean(rec_lpips_list),
    #     'recovery_mse': np.mean(rec_mse_list),
    #     'recovery_psnr': np.mean(rec_psnr_list),

    #     'Face_detection': np.mean(Face_detection_list),
    #     'Bounding_box_distance': np.mean(Bounding_box_distance_list),
    #     'Landmark_distance': np.mean(Landmark_distance_list),
    #     'arc_cos_similarity': np.mean(arc_cos_similarity_list),
    #     'vgg_cos_similarity': np.mean(vgg_cos_similarity_list),
    #     'casia_cos_similarity_list': np.mean(casia_cos_similarity_list),
    #     'anon_mse': np.mean(anon_mse_list),
    #     'anon_mae': np.mean(anon_mae_list),
    #     'arc_anon_mse': np.mean(arc_anon_mse_list),
    #     'arc_anon_mae': np.mean(arc_anon_mae_list),
    #     'vgg_anon_mse': np.mean(vgg_anon_mse_list),
    #     'vgg_anon_mae': np.mean(vgg_anon_mae_list),

    #     'fake_ssim': np.mean(fake_ssim_list),
    #     'fake_mse': np.mean(fake_mse_list),
    #     'fake_psnr': np.mean(fake_psnr_list),
    #     'fake_lpips': np.mean(fake_lpips_list),


    #     'Zfake_cos_similarity': np.mean(Zfake_cos_similarity_list),
    #     'fake2_cos_similarity': np.mean(fake2_cos_similarity_list),
    #     'Zrecon_cos_similarity': np.mean(Zrecon_cos_similarity_list),
    #     'fakeZ_cos_similarity': np.mean(fakeZ_cos_similarity_list),
        
    # }

    # metric_results = {
    #     # 'privacy_metric': np.mean(privacy_metric_list),
    #     'recovery_ssim': np.mean(rec_ssim_list),
    #     'recovery_lpips': np.mean(rec_lpips_list),
    #     'recovery_mse': np.mean(rec_mse_list),
    #     'recovery_psnr': np.mean(rec_psnr_list),
    #     'vgg_cos_similarity': np.mean(vgg_cos_similarity_list),
    #     'arc_cos_similarity': np.mean(arc_cos_similarity_list),
    #     'ada_cos_similarity': np.mean(ada_cos_similarity_list),
    #     'vgg_anon_mse': np.mean(vgg_anon_mse_list),
    #     'vgg_anon_mae': np.mean(vgg_anon_mae_list),
    #     'arc_anon_mse': np.mean(arc_anon_mse_list),
    #     'arc_anon_mae': np.mean(arc_anon_mae_list),
    #     'ada_anon_mae': np.mean(ada_anon_mae_list),
    #     'Face_detection': np.mean(Face_detection_list),
    #     'Bounding_box_distance': np.mean(Bounding_box_distance_list),
    #     'Landmark_distance': np.mean(Landmark_distance_list),
    #     'recon_arc_cos_similarity': np.mean(recon_arc_cos_similarity_list),

        
    # }

    # # print(f"****** {dname} ******")
    # print(metric_results)


if __name__ == '__main__':
    test_opts = TestOptions().parse()
    # if test_opts.test_batch_size != 1:
    #     raise Exception('This version only supports test batch size to be 1.')	
    # 设置随机数种子
    setup_seed(20)
    IMAGE_DATA_DIR = "/data/lk/ID-dise4/fake/small_image/"
    MASK_DIR = "/data/lk/ID-dise4/fake/small_mask/"
    W_DATA_DIR = "/data/lk/ID-dise4/fake/small_w/"
    run(test_opts, IMAGE_DATA_DIR, MASK_DIR, W_DATA_DIR)

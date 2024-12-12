import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from prettytable import PrettyTable
import logging
import torch
import net
from PPFR_model import ClientBackbone
from config import configurations
from util.utils import get_val_data, perform_val, get_test_celeA, perform_test_celebA

adaface_models = {
    'ir_50': "../model/adaface_ir50_webface4m.ckpt",
    'ir_101': "../model/adaface_ir101_webface12m.ckpt"
}


def load_AdaFace(architecture='ir_101'):
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture).cuda()
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()

    logging.info("AdaFace load success!")
    return model


# ClientBackbone
def load_model():
    cfg = configurations[1]
    BACKBONE = ClientBackbone(112, 512).cuda()
    backbone_resume = cfg['backbone_resume']
    backbone = torch.load(backbone_resume)

    BACKBONE.load_state_dict(backbone)
    BACKBONE.eval()

    return BACKBONE


if __name__ == '__main__':
    # ======= hyper parameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED']
    torch.manual_seed(SEED)
    DATA_ROOT = cfg['DATA_ROOT']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']
    BATCH_SIZE = cfg['BATCH_SIZE']

    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)

    # ----------------------------------Datasets-------------------------------------
    lfw, cfp_fp, agedb, calfw, cplfw, lfw_issame, cfp_fp_issame, agedb_issame, \
    calfw_issame, cplfw_issame = get_val_data(DATA_ROOT)

    # img_list, issame = get_test_celeA('../Data/CelebA/align_crop_112',
    #                                   '../Data/CelebA/Anno/triplet_loss/pairs.txt')

    BACKBONE = load_model()
    BACKBONE = BACKBONE.cuda()
    aux_model = load_AdaFace("ir_101")
    aux_model = aux_model.cuda()

    print("=" * 60)
    print("Perform Evaluation on LFW, CFP_FP, AgeDB, CALFW, CPLFW, CelebA_Test...")

    for i in range(4):
        if i == 0:
            print("Gaussian:")
        elif i == 1:
            print("Median:")
        elif i == 2:
            print("Pixelate:")
        elif i == 3:
            print("Hybrid:")

        blur_type = i

        # -------------------------celebA--------------------------------
        # accuracy_celebA, best_threshold_celebA, roc_curve_celebA = perform_test_celebA(EMBEDDING_SIZE, BATCH_SIZE,
        #                                                                                BACKBONE, img_list, issame,
        #                                                                                aux_model=aux_model,
        #                                                                                blur_type=blur_type)
        # --------------------------LFW-------------------------------
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(EMBEDDING_SIZE, BATCH_SIZE,
                                                                      BACKBONE, lfw, lfw_issame, aux_model=aux_model,
                                                                      blur_type=blur_type)
        # ---------------------------CFP-FP------------------------------
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(EMBEDDING_SIZE,
                                                                               BATCH_SIZE, BACKBONE, cfp_fp,
                                                                               cfp_fp_issame, aux_model=aux_model,
                                                                               blur_type=blur_type)
        # ---------------------------AgeDB------------------------------
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE, agedb, agedb_issame,
                                                                            aux_model=aux_model, blur_type=blur_type)
        # ----------------------------CALFW-----------------------------
        accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE,
                                                                            calfw, calfw_issame, aux_model=aux_model,
                                                                            blur_type=blur_type)
        # -----------------------------CPLFW----------------------------
        accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE,
                                                                            cplfw, cplfw_issame, aux_model=aux_model,
                                                                            blur_type=blur_type)

        Average = (accuracy_agedb + accuracy_calfw + accuracy_cfp_fp + accuracy_cplfw + accuracy_lfw) / 5
        accu_list = [accuracy_agedb, best_threshold_agedb, accuracy_calfw, best_threshold_calfw,
                     accuracy_cfp_fp, best_threshold_cfp_fp, accuracy_cplfw, best_threshold_cplfw,
                     accuracy_lfw, best_threshold_lfw, Average]
        name = ["Agedb_acc", "Agedb_threshold", "CALFW_acc", "CALFW_threshold",
                "CFP_FP_acc", "CFP_FP_threshold", "CPLFW_acc", "CPLFW_threshold",
                "LFW_acc", "LFW_threshold", "Average"]
        pretty_tabel = PrettyTable(["name", "result_AdaFace"])
        index = 0
        for temp in accu_list:
            pretty_tabel.add_row([name[index], temp])
            index += 1
        print(pretty_tabel)

        # print("celebA_Test_acc:{:.4f} \ncelebA_Test_threshold:{:.4f}".format(accuracy_celebA, best_threshold_celebA))
        # print("=" * 60)

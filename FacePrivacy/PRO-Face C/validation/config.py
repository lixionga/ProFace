
configurations = {
    1: dict(
        SEED=1234,

        DATA_ROOT='../Data/validation',  # To place the validation dataset file

        INPUT_SIZE=[112, 112],
        EMBEDDING_SIZE=512,
        BATCH_SIZE=128,

        backbone_resume='../model/ckpt/Backbone.pth',  # place the pre-trained model for testing

    ),
}

def select_network(opt):
    # network = opt['network']['name']
    network = 'network_basic'
    if network == 'network_basic':
        from networks.network_basic import ResTransformer
    # elif network == 'network_pooling':
    #     from networks.network_pooling import ResTransformer
    # elif network == 'network_maxpooling':
    #     from networks.network_maxpooling import ResTransformer
    # elif network == 'network_vit':
    #     from networks.network_vit import ResTransformer
    # elif network == 'network_noT':
    #     from networks.network_noT import ResTransformer
    # elif network == 'network_gate':
    #     from networks.network_gate import ResTransformer
    # elif network == 'network_gate2':
    #     from networks.network_gate2 import ResTransformer
    # elif network == 'network_conv':
    #     from networks.network_conv import ResTransformer
    # elif network == 'network_unet':
    #     from networks.network_unet import ResTransformer
    return ResTransformer()


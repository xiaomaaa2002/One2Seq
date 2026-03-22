__all__ = ['build_encoder']


def build_encoder(config):
    from .resnet31_rnn import ResNet_ASTER
    from .svtrnet import SVTRNet
    from .svtrnet2dpos import SVTRNet2DPos
    from .svtrv2 import SVTRv2
    from .svtrv2_lnconv import SVTRv2LNConv
    from .svtrv2_lnconv_two33 import SVTRv2LNConvTwo33
    from .vit import ViT
    from .clip_encoder import CLIPEncoder
    
    support_dict = [
        'SVTRNet', 'ViT',
        'SVTRNet2DPos', 'SVTRv2', 'ResNet_ASTER',
        'SVTRv2LNConv', 'SVTRv2LNConvTwo33', 'CLIPEncoder',
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when encoder of rec model only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class

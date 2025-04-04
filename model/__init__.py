from .kdgraph import KDGraph

def build_model(in_ch, model_key='', backbone='resnet50', pretrained_flag=True):
    if model_key == "KDGraph":
        model = KDGraph(in_ch, backbone, pretrained_flag)
    else:
        ValueError("Error model name")
    return model

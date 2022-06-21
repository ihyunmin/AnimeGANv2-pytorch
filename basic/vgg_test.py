#
# This code can test the VGG19-batch normalization model. It is a test for content loss.
#

import torch
from vgg.vgg import VGG

# It isn't able to control the extract layer of the vgg model.
# It is from the previous layer of fully connected layer.
def main():

    # model load
    model_name = 'vgg19'
    model = VGG.from_pretrained(model_name)
    print(model)

    # model inference test
    sample_input = torch.rand((12,3,224,224))
    out = model(sample_input)
    features = model.extract_features(sample_input)
    # print(out)
    # out -> torch.Size([b, 1000])
    print(out.shape)
    
    # features -> torch.Size([b, 512, 7, 7])
    print(features.shape)

if __name__=="__main__":
    main()
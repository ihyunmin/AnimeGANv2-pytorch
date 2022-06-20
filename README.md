# AnimeGANv2 Pytorch version include train code.        

## Summary           

This repository includes train code of the AnimeGANv2, pytorch version.       

There are three types of train code, pytorch version(port original(tensorflow) to torch),       

train for Dragonball, and train for Polygon art). I had to seperate versions to adapt their datasets.       

However, I can't share to github about dragonball version and polygon art version, because the idea belongs to klleon.    

Overall, the structure of this repository is below.         

## Structure of this

### Folders        

checkpoint - If you start train.py, the model parameters are saved to .pth files every the end of epoch.         

dataset - There are many style datasets, and they have two types of data, smooth and style.         

google_images_download - If you need more pictures, you can get images from this folder, but you need chromedriver which is same to your chrome version.           

model - There are discriminator and generator, generator was from Ref 2 made by bryandlee, and discriminator was ported by me.          

runs - If you use tensorflow and command "tensorboard --logdirs ./runs", It will be saved in this folder.         

samples - If you start train.py, the validation sets are saved in this folder.        

tools - There are data loader and some util python codes about preparing the images.           

vgg - There are vgg-19 model to use content loss and style loss.        

### Files            

AnimeGANv2.py - There is a class which can train the generator and discriminator.           

edge_smooth.py - If you have the style images, this can make smooth images by using Gaussian filter.         

losses.py - Loss terms for traning the model.      

resize_pictures.py - This model must use 256x256 pictures to train, so it can make 256x256 size of the datasets.       

server_output_download.sh - If you want to get the result in the server, bash server_output_download.sh.         

tensor_board.py - It is tensorboard class. I want to make the train code short and efficient.        

tensor_board_test.py - To study the tensorboard in Pytorch.        

test.py - I did something to test Pytorch.         

vgg_test.py - I did something to test vgg.         

### Note

I usually train the model at the cafe24 server, and I receive the results by using "scp command", so the samples folder had a lot of results. Moreover, I should change the train method and loss weight for each datasets, so this repository includes three types of code.         

If you want to know the details of this repository, see notion page.     

https://www.notion.so/klleon/b1ad029a28724d10b182ffadc2928acc          


## Reference      

1. Original AnimeGANv2 - https://github.com/TachibanaYoshino/AnimeGANv2          

2. AnimeGANv2-pytorch - bryandlee - https://github.com/bryandlee/animegan2-pytorch        

3. Neural Style Transfer - https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/neural_style_transfer         

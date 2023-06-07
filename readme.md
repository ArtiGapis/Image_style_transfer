# Image style transfer

You can find this web app image transfer on this page
http://arturasportfolio.website/

### About
This project is a Python script for transferring the rendered style from one image to 
another and moving the rendering to HTML for integration into any type of website. 
The script defines several functions for loading and processing images, 
defining style and content layers, creating a style transfer model, calculating loss, 
and optimizing the generated image. This script is a convenient way to transfer style 
using the VGG19 model and TensorFlow. To use this script, you can provide content and 
select a style image from the list, specify the desired number of epochs.

I attached several examples with the most optimal weights, learning rate, layers and
after choosing to learn 50 epochs. Using these settings, we get the most optimal result, the image of the photo is preserved and intact.<br/>

<img src="images/cropped_drawing_mod.jpg" height="150" alt=""/>
<img src="images/grey_pencil_style_mod.jpg" height="150" alt=""/>
<img src="images/hole_in_the_stone_mod.jpg" height="150" alt=""/>
<img src="images/van_gogh_style_mod.jpg" height="150" alt=""/>
<img src="images/watercolor_style_mod.jpg" height="150" alt=""/>


### Introduction
In the previous model, similar results were achieved in pire 1000 epochs, which took about 3 hours.
Currently, after selecting model VGG19 by trial adjustment learning_rate, content_weight, style_weight
the result is obtained in a few minutes.
1. Increased content weight to keep more of the original photo.
2. The weight of the style is reduced to control the relative importance of the content.
3. Reduced learning rate by two times.

After choosing a photo of 224 height, 50 epochs, it trains for about 3 minutes, choosing
for a larger photo, the time increases significantly, so it is advisable to use a height of 224 for testing.

### Startup procedure

You should run a command line.
You should write a "git clone https://github.com/ArtiGapis/Image_style_transfer.git" in a command line.
Open project in your favorite editor / IDE which use a Python v3.10.
Install the necessary libraries: 'pip install -r requirements.txt'
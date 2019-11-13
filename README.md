# Sports-Video-Classifier
A Deep Learning Project which classifies various videos into 10 different classes.
Uses Transfer Learning from Inception v3 Model.

Data Set: http://cvlab.cse.msu.edu/project-svw.html

# General Introduction and the Idea: 

A quick glance at an image is sufficient for a human to point out and describe immense amount of details about the visual scene. However, this remarkable ability has proven to be a difficult task to reciprocate on machines. What if machines become so smart that they provide us exactly what we ask from them? Everywhere in this world researchers are trying to reach this level of accuracy. This is a fact that there is a lot of video data available on the internet and undoubtedly videos are the best way to share and understand knowledge, but like every other thing they too have drawbacks. Since a video contains a large amount of information finding exact instance of any particular topic becomes tedious. So, we are trying to make videos searchable through content, which will search for particular topic within video and return the exact instance of it in the video. Also, it will help us to deal with another major problem of plagiarism and the famous click bait issue on YouTube.

YouTube has 400 hours of video uploaded every minute. One of the most difficult tasks that users face on such sites is to find the interesting/relevant videos from the search results without opening and going through each one. While video summarization seems like a nice solution to the problem, a better one would be to make videos searchable for their content via Video Captioning, so that one might search for the part of the video that they are interested in rather than spending time watching on all of it. When we summarize a video by reducing the duration of the video, we lose a lot of information. This maybe especially bad for content where sound is as informative as the video frames. This is because during video summarization, we skip frames making intermittent breaks in the audio playback. However, if we make videos searchable by content, we might just be able to skip through all the unnecessary details and watch the part that is important to us in all of its detail. In this project we present a tool to arts of videos that can then be searched for.

# Working

MODULE I- PREPROCESSING

1. I had a Sports Dataset (SVW) from which I chose 10 different sports namely Pole vault, Baseball, Soccer, BMX, Skiing, Rowing, Golf, Tennis, Long jump and Hurdling.
2. I chose 50 videos of each category and saved them separately in the folders named after the sport: which will later act as our labels.
3. Then, videos were converted into frames which resulted in a very large number of frames also the number of frame were inconsistent among the classes.
4. I improved by comparing each frame by preceding one and if the similarity index was less than a threshold value (0.7 at the start).
5. Around 500 of frames were used for training and 150 for testing purpose for each class.

MODULE II- CNN

1. After converting videos to frames, we make tf records and feed the data to input pipeline joining our CNN.
2. We have coloured images of size 100*100.
3. First layer is a convolutional layer, of size 5*5 pixels which convolutes over the image and applies 16 filters.
4. It is followed by a max pool layer which has a stride of 2. Therefore, it down samples the image to 50*50 pixels.
5. Again, a convolutional layer is applied to the image but this time extracting 36 filters.
6. Both the convolutional layers are followed by ReLU activation function which sets all the negative values to zero.
7. Then again a max pooling layer if 2 strides is applied which down samples the image to 25*25 pixels.
8. This output acts as an input for the next fully connected layer, this layer converts the input image to one dimensional array of values and also all the neurons are connected to each other.
9. Finally, another FCC layer is applied but this time its output is 10 classes.
10. When we input a validation video, it is converted into frames and they are fed into this model.
11. The output of this model is 10*1 matrix with probabilities of each label- the six classes.
12. Then we use argmax function to extract the highest value and this value is predicted as a class.

MODULE III- TRANSFER LEARNING

1. After making Data Directory which contains all the images for training and testing in appropriate folder structure, I stored both the images and their labels.
2. After that I downloaded Inception v3 model directly using the code.
3. The images are of size 100*100 pixels and are coloured.
4. Then these images are fed into the Inception model which is pre-trained on millions of different images and has several classes.
5. Aim is to extract transfer values of the images and save them into a cache file.
6. After that I have added 2 layers of my own in front of Inception model.
7. First one is a Fully Connected Layer which is takes input an array of 1D images of 1024 pixels and provides confidence into 10 classes.
8. After this a softmax layer is used to make the values under 1.
9. Finally argmax function is used to provide the highest confidence value class.



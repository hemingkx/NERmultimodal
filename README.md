# NERmultimodal

## 1. Introduction

(unofficial) Keras ReImplementation of "Adaptive Co-attention Network for Named Entity Recognition in Tweets".  The original code is in [NERmultimodal](https://github.com/jlfu/NERmultimodal) .

## 2. Requirements

1. Python 2.7 or higher
2. Keras 1.2, the backend is theano.
3. The image features were extracted from 16-layer VGGNet. Before extracting the features, you need to download a pretrained model -- [vgg16_weights_th_dim_ordering_th_kernels_notop.h5](https://github.com/fchollet/deep-learning-models/releases).
4. Moreover, you need to download the word embedding trained by tweets from http://pan.baidu.com/s/1boSlljL (which is slow to be downloaded, you can only download word2vec_200dim.model in /word2vec path).

## 3. Get Started

To get the right environment settled, you can create an anaconda vitual env by

```
conda create -n env_name python=2.7
```

Then install these packages one by one

```
pip install keras==1.2.2
pip install h5py
pip install pip==9.0.1
pip install gensim==3.8.3
conda install mkl-service
```

ps: pip 9.0.1 is used to install gensim 3.8.3

All packages that are needed to run the code can be seen in requirements.txt (not including the feature extraction part).

For feature extraction, you need to install opencv by

```
pip install opencv-python==3.1.0.0
```

## 4. Details

I rewrite a few parts in the original code to make it run. 

In **vgg_image_feature.py** line 85, I change the original code to

```
for text_filename in ['train', 'dev', 'test']:
    with open(os.path.join('./data/', text_filename), 'r') as f:
        for line in f:
            if line.startswith("IMGID:"):
                img_id_list.append(line.replace("IMGID:", "").strip())
```

In **multimodal_ner.py**, I do not use the CRF from keras.layer, but use the CRF implementation from [emnlp2017-bilstm-cnn-crf](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf) , which can be downloaded [here](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/releases/tag/1.2.2) . Its CRF implementation file is in the /neuralnets/keraslayers path.

After set all the requirements above, you can run the code smoothly.

## 5. More Information 

For more Information, please check [the original code](https://github.com/jlfu/NERmultimodal) . 
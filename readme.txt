deep-person-reid

link:https://github.com/KaiyangZhou/deep-person-reid
paper:Li_Harmonious_Attention_Network_CVPR_2018_paper
这次主要分析HA-CNN的神经网络和deep-person-reid的文件组织，对于可有可无的，适当删减

.
├── BENCHMARK.md
├── DATASETS.md
├── imgs
│   └── ranked_results.jpg
├── LICENSE
├── README.md
├── RELATED_PROJECTS.md
├── requirements.txt
├── torchreid
│   ├── data_manager  # data_manager只提供dataset的路径，与dataset_loader配合
│   │   ├── cuhk01.py
│   │   ├── cuhk03.py
│   │   ├── dukemtmcreid.py
│   │   ├── dukemtmcvidreid.py
│   │   ├── grid.py
│   │   ├── ilids.py
│   │   ├── ilidsvid.py
│   │   ├── __init__.py
│   │   ├── market1501.py
│   │   ├── mars.py
│   │   ├── msmt17.py
│   │   ├── prid2011.py
│   │   ├── prid450s.py
│   │   ├── sensereid.py
│   │   └── viper.py
│   ├── dataset_loader.py # 与data_manager配合，提供__getitem__和len
│   ├── eval_lib
│   │   ├── eval.pyx
│   │   ├── __init__.py
│   │   ├── Makefile
│   │   ├── setup.py
│   │   └── test_cython_eval.py
│   ├── eval_metrics.py  # CMC和mAP,用的是欧式距离
│   ├── __init__.py 
│   ├── losses       # 作者定义了几种loss，
│   │   ├── center_loss.py
│   │   ├── cross_entropy_loss.py
│   │   ├── hard_mine_triplet_loss.py
│   │   ├── __init__.py
│   │   └── ring_loss.py
│   ├── models
│   │   ├── densenet.py
│   │   ├── hacnn.py
│   │   ├── inceptionresnetv2.py
│   │   ├── inceptionv4.py
│   │   ├── __init__.py    # 提供模型的统一接口，与CycleGAN比起来，没有那么智能，但是看起来更简洁，更方便理解。
│   │   ├── mobilenetv2.py
│   │   ├── mudeep.py
│   │   ├── nasnet.py
│   │   ├── resnet.py
│   │   ├── resnext.py
│   │   ├── seresnet.py
│   │   ├── shufflenet.py
│   │   ├── squeeze.py
│   │   └── xception.py
│   ├── optimizers.py   # 提供optimizer
│   ├── samplers.py
│   ├── transforms.py
│   └── utils
│       ├── avgmeter.py
│       ├── __init__.py
│       ├── iotools.py
│       ├── logger.py
│       ├── reidtools.py
│       └── torchtools.py
├── train_imgreid_xent_htri.py
├── train_imgreid_xent.py
├── train_vidreid_xent_htri.py
└── train_vidreid_xent.py

***************************************************************************
现在有一个疑问，就是train和test的model是一样的，为什么一个是outputs，一个是feature

***************************************************************************
网络
/home/tjj/anaconda3/bin/python "/media/tjj/Document/1Paper/8 deep-person-reid/torchreid/models/hacnn.py"

InceptionA(
      (stream1): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream2): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream3): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream4): Sequential(
        (0): AvgPool2d(kernel_size=3, stride=1, padding=1)
        (1): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
InceptionB(
      (stream1): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream2): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): ConvBlock(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream3): Sequential(
        (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (1): ConvBlock(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )


HACNN(
  (conv): ConvBlock(
    (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (inception1): Sequential(
    )
  (ha1): HarmAttn(
    (soft_attn): SoftAttn(
      (spatial_attn): SpatialAttn(
        (conv1): ConvBlock(
          (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): ConvBlock(
          (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (channel_attn): ChannelAttn(
        (conv1): ConvBlock(
          (conv): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): ConvBlock(
          (conv): Conv2d(8, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (conv): ConvBlock(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (hard_attn): HardAttn(
      (fc): Linear(in_features=128, out_features=8, bias=True)
    )
  )
  (inception2): Sequential(
    (0): InceptionA(
      (stream1): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream2): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream3): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream4): Sequential(
        (0): AvgPool2d(kernel_size=3, stride=1, padding=1)
        (1): ConvBlock(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (1): InceptionB(
      (stream1): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream2): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): ConvBlock(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream3): Sequential(
        (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (1): ConvBlock(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
  )
  (ha2): HarmAttn(
    (soft_attn): SoftAttn(
      (spatial_attn): SpatialAttn(
        (conv1): ConvBlock(
          (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): ConvBlock(
          (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (channel_attn): ChannelAttn(
        (conv1): ConvBlock(
          (conv): Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): ConvBlock(
          (conv): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (conv): ConvBlock(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (hard_attn): HardAttn(
      (fc): Linear(in_features=256, out_features=8, bias=True)
    )
  )
  (inception3): Sequential(
    (0): InceptionA(
      (stream1): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(256, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream2): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(256, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream3): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(256, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream4): Sequential(
        (0): AvgPool2d(kernel_size=3, stride=1, padding=1)
        (1): ConvBlock(
          (conv): Conv2d(256, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (1): InceptionB(
      (stream1): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream2): Sequential(
        (0): ConvBlock(
          (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ConvBlock(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): ConvBlock(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (stream3): Sequential(
        (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (1): ConvBlock(
          (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
  )
  (ha3): HarmAttn(
    (soft_attn): SoftAttn(
      (spatial_attn): SpatialAttn(
        (conv1): ConvBlock(
          (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): ConvBlock(
          (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (channel_attn): ChannelAttn(
        (conv1): ConvBlock(
          (conv): Conv2d(384, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): ConvBlock(
          (conv): Conv2d(24, 384, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (conv): ConvBlock(
        (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (hard_attn): HardAttn(
      (fc): Linear(in_features=384, out_features=8, bias=True)
    )
  )
  (fc_global): Sequential(
    (0): Linear(in_features=384, out_features=512, bias=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (classifier_global): Linear(in_features=512, out_features=751, bias=True)
  (local_conv1): InceptionB(
    (stream1): Sequential(
      (0): ConvBlock(
        (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): ConvBlock(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stream2): Sequential(
      (0): ConvBlock(
        (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): ConvBlock(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): ConvBlock(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stream3): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (1): ConvBlock(
        (conv): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (local_conv2): InceptionB(
    (stream1): Sequential(
      (0): ConvBlock(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): ConvBlock(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stream2): Sequential(
      (0): ConvBlock(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): ConvBlock(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): ConvBlock(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stream3): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (1): ConvBlock(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (local_conv3): InceptionB(
    (stream1): Sequential(
      (0): ConvBlock(
        (conv): Conv2d(256, 96, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): ConvBlock(
        (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stream2): Sequential(
      (0): ConvBlock(
        (conv): Conv2d(256, 96, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): ConvBlock(
        (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): ConvBlock(
        (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stream3): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (1): ConvBlock(
        (conv): Conv2d(256, 192, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (fc_local): Sequential(
    (0): Linear(in_features=1536, out_features=512, bias=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (classifier_local): Linear(in_features=512, out_features=751, bias=True)
)

Process finished with exit code 0


***************************************************************************
重定向
sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

Logger.py

from __future__ import absolute_import

import sys
import os
import os.path as osp

from .iotools import mkdir_if_missing


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
***************************************************************************
自己对于model.__class__.__name__一直很陌生，这一块要加强一下才行，之前遇到也都避过了。
def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
***************************************************************************
模型参数个数，应该看看
def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param
***************************************************************************
在parser中是label-smooth,在调用的时候就变成了label_smooth
parser.add_argument('--label-smooth', action='store_true',help="use label smoothing regularizer in cross entropy loss"
arg.label_smooth

***************************************************************************
torch.scatter_

from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    
    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

scatter_的用法
scatter_(dim, index, src) 
以dim=0为例
用个图片好了，写不出来

>>> x = torch.rand(2, 5)
>>> x
tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
        [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
>>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
        [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
        [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])
***************************************************************************
多GPU运行torch
model
model2 = nn.DataParallel(model1)
print(model2)
for var in model2.parameters():
    print(var)

DataParallel(
  (module): Model(
    (fc): Linear(in_features=3, out_features=4, bias=True)
  )
)
Parameter containing:
tensor([[ 0.1964,  0.4389, -0.2216],
        [-0.1046, -0.2055, -0.5383],
        [ 0.0673,  0.0949,  0.5205],
        [ 0.5473, -0.3700, -0.4179]], device='cuda:0')
Parameter containing:
tensor([ 0.2416,  0.4188, -0.0096,  0.1569], device='cuda:0')

model2.cuda()

不是很理解，为什么在DataParallel之后还要加一个cuda
https://www.jianshu.com/p/0bdf846dc1a2

***************************************************************************
model.apply()
model.apply(set_bn_to_eval)
def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
***************************************************************************
eval_market1501
eval_cuhk03
的公式不一样，有时间的话就实践一下各自是怎么计算的
因为感觉每次算CMC和mAP 都似乎很麻烦的样子
***************************************************************************
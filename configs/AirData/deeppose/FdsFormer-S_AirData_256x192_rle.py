_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/datasets/Air.py'
]
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)

optimizer_config = dict(grad_clip=None)
# learning policy
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[5, 7])
total_epochs = 10
channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3)),
    neck=dict(
        type='FdsFormer',
        in_channels=[256,512,1024,2048],
        embed_dims=[256,512,1024,2048],
        num_layers=[1, 2, 4, 2],
        return_map=True
    ),
    keypoint_head=dict(
        type='DeepposeRegressionHead',
        in_channels=2048,
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(
            type='RLELoss',
            use_target_weight=True,
            size_average=True,
            residual=True),
        out_sigma=True),
    train_cfg=dict(),
    test_cfg=dict(flip_test=True))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/AirData/aircraft_detection_results/'
    'Air_val_detections.json',
    # bbox_file = ''
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    # dict(
    #     type='TopDownHalfBodyTransform',
    #     num_joints_half_body=8,
    #     prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetRegression'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/AirData'
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=12),
    test_dataloader=dict(samples_per_gpu=12),
    train=dict(
        type='AirDataset',
        ann_file=f'{data_root}/annotations/keypoints_train.json',
        img_prefix=f'{data_root}/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='AirDataset',
        ann_file=f'{data_root}/annotations/keypoints_val.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='AirDataset',
        ann_file=f'{data_root}/annotations/keypoints_val.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

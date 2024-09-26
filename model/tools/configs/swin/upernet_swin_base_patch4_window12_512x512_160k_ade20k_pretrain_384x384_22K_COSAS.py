_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/COSAS.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
class_count = 2
model = dict(
        pretrained='/home/cai_user/LightM-Unet/mm/pretrain/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth',
    backbone=dict(
        pretrain_img_size=512,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
        decode_head=dict(#ignore_index=0,
            #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
            in_channels=[128, 256, 512, 1024], num_classes=class_count,
            loss_decode=[
                        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]),

        auxiliary_head=dict(#ignore_index=0,
            #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
            in_channels=512, num_classes=class_count,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]))

data = dict(samples_per_gpu=4)

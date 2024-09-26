# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', warmup='linear', warmup_iters=500, 
        warmup_ratio=1e-5, 
        power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
#runner = dict(type='IterBasedRunner', max_iters=80000)
#checkpoint_config = dict(by_epoch=False, interval=2000)
#evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
runner = dict(type='EpochBasedRunner', max_epochs=100) 
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(interval=1, metric=['mIoU', 'mDice'], save_best='mIoU', pre_eval=True)

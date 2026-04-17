# 勿 `from datetime import datetime`：会把 datetime 并入 cfg，MMEngine pretty_text/YAPF 会语法错误崩溃。

_base_ = ['./mask2former_r50_8xb2-160k_ade20k-512x512.py']

# 预训练权重：相对「当前工作目录」解析。请勿在 cfg 里用 __file__：MMEngine 合并 _base_ 时 exec 环境无 __file__。
# 必须在 mask2former 目录下运行 train.py / test.py，使路径指向 mask2former/data/checkpoints1/xxx.pth。
# 权重与预训练同目录：work_dir == checkpoint.out_dir 时 MMEngine 不会再套一层子文件夹。
# 每次启动训练会生成新的时间戳前缀，便于和官方 .pth 区分；best 仍由 save_best 生成 best_mIoU_*.pth。
_ckpt_time = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = 'data/checkpoints1'
load_from = 'data/checkpoints1/mask2former_transnext_tiny_512x512_160k_ade20k_in1k.pth'
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        save_best='mIoU',
        out_dir='data/checkpoints1',
        filename_tmpl=(
            f'finetune_transnext_tiny_160k_ade20k_{_ckpt_time}_iter{{}}.pth'),
        save_last=True))

depths = [2, 2, 15, 2]
model = dict(
    backbone=dict(
        type='transnext_tiny',
        pretrain_size=224,
        img_size=512,
        pretrained=None),
    decode_head=dict(in_channels=[72, 144, 288, 576]))


backbone_nodecay = dict(lr_mult=0.1, decay_mult=0)
backbone_decay = dict(lr_mult=0.1)
embed_mult = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {'attn.query_embedding': backbone_nodecay,
               'relative_pos_bias_local': backbone_nodecay,
               'cpb': backbone_nodecay,
               'temperature': backbone_nodecay,
               'attn.learnable': backbone_decay,
               'attn.q.weight': backbone_decay,
               'attn.q.bias': backbone_nodecay,
               'attn.kv.weight': backbone_decay,
               'attn.kv.bias': backbone_nodecay,
               'attn.qkv.weight': backbone_decay,
               'attn.qkv.bias': backbone_nodecay,
               'attn.sr.weight': backbone_decay,
               'attn.sr.bias': backbone_nodecay,
               'attn.norm': backbone_nodecay,
               'attn.proj.weight': backbone_decay,
               'attn.proj.bias': backbone_nodecay,
               'mlp.fc1.weight': backbone_decay,
               'mlp.fc2.weight': backbone_decay,
               'mlp.fc1.bias': backbone_nodecay,
               'mlp.fc2.bias': backbone_nodecay,
               'mlp.dwconv.dwconv.weight': backbone_decay,
               'mlp.dwconv.dwconv.bias': backbone_nodecay,
               'decode_head.query_embed': embed_mult,
               'decode_head.query_feat': embed_mult,
               'decode_head.level_embed': embed_mult
               }
custom_keys.update({
    f'backbone.norm{stage_id + 1}': backbone_nodecay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id + 1}.norm': backbone_nodecay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id + 1}.proj.weight': backbone_decay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id + 1}.proj.bias': backbone_nodecay
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'backbone.block{stage_id + 1}.{block_id}.norm': backbone_nodecay
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, bias_decay_mult=0, norm_decay_mult=0, flat_decay_mult=0),
    #accumulative_counts=2,
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# 训练轮次、batch、LR 调度、checkpoint 间隔等与官方 Mask2Former+ADE20K 一致，见父配置：
# mask2former_r50_8xb2-160k_ade20k-512x512.py（160k iter，val/ckpt 每 5000 iter）。
# 若需短跑调试，请改用 mask2former_transnext_tiny_quick_ade20k-512x512.py。

import shuffled_mae_case2


model = shuffled_mae.__dict__['mae_vit_large_patch16_dec512d8b'](norm_pix_loss=True)
print(model.parameters())
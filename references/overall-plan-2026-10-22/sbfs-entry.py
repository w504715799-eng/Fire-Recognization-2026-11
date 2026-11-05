
import torch
import torch.nn as nn
import torch.nn.functional as F
from ASNET_encoder import MixVisionTransformer
from ASNET_decoder_v1_ab00 import asnet_decoder

class asnet(nn.Module):
	def __init__(self, 
			  	 img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
				 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.01,
				 attn_drop_rate=0.01, drop_path_rate=0.01, norm_layer=nn.LayerNorm,
				 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
				 
				 feats_channels=[64,128,320,512], d_depths = [2,2,3,2],
				 sb_drop_rate=0.01, cnn_drop_rate=0.01,seg_drop_rate=0.5,

				 pretrained_encoder = None,#'./mit_b3.pth',
				 pretrained = None,#'./pth/asnet_syn70k_v3_set1_epoch42.pth',
				 mode = 'train',
				 branch = 'seg'
				):
		super().__init__()

		self.encoder = MixVisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dims,
				 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
				 attn_drop_rate, drop_path_rate, norm_layer,
				 depths, sr_ratios, mode = mode)
		self.decoder = asnet_decoder(feats_channels,d_depths,
							   sb_drop_rate, cnn_drop_rate,seg_drop_rate,
							   mode = mode, branch= branch  )
		self.mode = mode

		if pretrained != None:
			#以路径形式载入
			pre_dict = torch.load(pretrained)
			self_dict = self.state_dict()
			updateflag = False
			for k,v in pre_dict.items():
				if k in self_dict:
					if v.shape == self_dict[k].shape:
						self_dict.update({k:v})
						updateflag = True
				elif k.replace('module.','') in self_dict:
					if v.shape == self_dict[k.replace('module.','')].shape:
						self_dict.update({k.replace('module.',''):v})
						updateflag = True
			#self_dict.update({k:v for k,v in pre_dict.items() if k in self_dict})
			self.load_state_dict(self_dict)
			print(f'Update pretrained dict: {updateflag}')
		elif pretrained_encoder != None:
			pre_e_dict = torch.load(pretrained_encoder)
			self_e_dict = self.encoder.state_dict()
			updateflag = False
			for k,v in pre_e_dict.items():
				if k in self_e_dict:
					self_e_dict.update({k:v})
					updateflag = True
			self.encoder.load_state_dict(self_e_dict)
			print(f'Update encoder pretrained dict: {updateflag}')

	def forward(self,x):
		
		if self.mode == 'train':
			feats = self.encoder(x)
			# seg, alpha = self.decoder(feats)
			# return seg, alpha
			seg = self.decoder(feats)
			return seg
		else:
			feats = self.encoder(x)
			seg, alpha , pics = self.decoder(feats)
			return seg, alpha,  pics

if __name__ == '__main__':
	im = torch.ones(1,3,256,256)
	model = asnet()
	seg, alpha = model(im)
	print(seg.shape)
	print(alpha.shape)
	#print(model.state_dict().keys())
	'''with open('./log/model_drop.log','w') as f:
		for k in model.state_dict().keys():
			f.write(f'{k}\n')
	with open('./log/v2_22_pth.log','w') as f:
		pthdict = torch.load('./pth/asnet_syn70k_TrainOn70k_v2_22_new.pth')
		for k in pthdict.keys():
			f.write(f'{k}\n')'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F

class BoxCoder(nn.Module):
	def __init__(self, patch_count, patch_stride, patch_size, seq_len, channels, device='cuda:0'):
		super().__init__()
		self.device = device

		self.seq_len = seq_len
		self.channels = channels
		self.patch_size = patch_size
		self.patch_count = patch_count
		# print("patch_count{patch_count}")
		self.patch_stride = patch_stride
		
		self._generate_anchor(device=device)
		
	# compute the center points. idx: [0 ~ seq_len - 1]
	def _generate_anchor(self, device="cuda:0"):
		anchors = []
		self.S_bias = (self.patch_size - 1) / 2
		print(f"anchor patch_count2: {self.patch_count}")
		for i in range(self.patch_count):
			x = i * self.patch_stride + 0.5 * (self.patch_size - 1)
			anchors.append(x)

		anchors = torch.as_tensor(anchors, device=device)
		self.register_buffer("anchor", anchors)

	def forward(self, boxes):
		#(B, patch_count, channels, 2)
		# print(f"boxes.shape: {boxes.shape}") 
		self.bound = self.decode(boxes) # (bs, patch_count, channel, 2)->(B, patch_count, channels, 2) 
		points = self.meshgrid(self.bound) #(B, patch_count, channels, 2)->(B, patch_count, channels, patch_size, 2)
		#->points: 采样网格坐标，形状 (B, patch_count, channels, patch_size, 2)
		#  bound: 归一化的分块边界，形状 (B, patch_count, channels, 2)
		return points, self.bound

	def decode(self, rel_codes):  # Input: (B, patch_count, channel, 2)  128 12 1 2
		# print(f"rel_codes.shape: {rel_codes.shape}")
		boxes = self.anchor #6
		
		dx = rel_codes[:, :, :, 0]
  
		# print(f"dx.shape: {dx.shape}") #[128, 12, 1]
		ds = torch.relu(rel_codes[:, :, :, 1] + self.S_bias)
		# print(f"ds.shape: {ds.shape}") #[128, 12, 1]
		#创建一个与 rel_codes 形状相同的全零张量。
		pred_boxes = torch.zeros_like(rel_codes) #128 12 1 2

  
  
  
		# print(f"pred_boxes.shape: {pred_boxes.shape}")
		# print(f"boxes.shape: {boxes.shape}")
		#将锚点 boxes 从 (P,) 调整为 (1, P, 1)。 1 6 1
		ref_x = boxes.view(1, boxes.shape[0], 1)
		# print(f"ref_x.shape: {ref_x.shape}")
		# print(f"dx.shape: {dx.shape}")

		# dx, ds: (bs, patch_count, channel, 1)
		# ref_x 的维度 (1, P, 1) 会被广播为 (B, P, C)。
		# dx 和 ds 的维度均为 (B, P, C)
		pred_boxes[:, :, :, 0] = (dx + ref_x - ds) # 左边界
		pred_boxes[:, :, :, 1] = (dx + ref_x + ds) # 右边界
  
  
		# 归一化到 [0, 1] 范围
		pred_boxes /= (self.seq_len - 1)
		# 确保所有值在 [0, 1] 范围内。
		pred_boxes = pred_boxes.clamp_(min=0., max=1.)

		# pred_boxes: each of the patch's left-bound & right-bound. norm to [0, 1] (B, P, C, 2)
		# max_right = (self.seq_len - 1)  # 最大允许的右边界索引
		# pred_boxes[:, :, :, 0] = torch.clamp(pred_boxes[:, :, :, 0] * (self.seq_len - 1), min=0, max=max_right) / (self.seq_len - 1)
		# pred_boxes[:, :, :, 1] = torch.clamp(pred_boxes[:, :, :, 1] * (self.seq_len - 1), min=0, max=max_right) / (self.seq_len - 1)
		return pred_boxes	
   
	def meshgrid(self, boxes): # Input: pred_boxes. To get the sampling location
		B, patch_count, C = boxes.shape[0], boxes.shape[1], boxes.shape[2]
		channel_boxes = torch.zeros((boxes.shape[0], boxes.shape[1], 2)).to(self.device)
		channel_boxes[:, :, 1] = 1.0
		xs = boxes.view(B*patch_count, C, 2)
		xs = torch.nn.functional.interpolate(xs, size=self.patch_size, mode='linear', align_corners=True)
		ys = torch.nn.functional.interpolate(channel_boxes, size=self.channels, mode='linear', align_corners=True)

		# xs: [bs, patch_count, channel, patch_size]   ys: [bs, patch_count, channels(also feats)]
  
		xs = xs.view(B, patch_count, C, self.patch_size, 1)
		ys = ys.unsqueeze(3).expand(B, patch_count, C, self.patch_size).unsqueeze(-1)
  
		grid = torch.stack([xs, ys], dim = -1)
		return grid # [bs, patch_count, channel, patch_size, 2]

def zero_init(m):
	if type(m) == nn.Linear or type(m) == nn.Conv1d:
		m.weight.data.fill_(0)
		m.bias.data.fill_(0)

class OffsetPredictor(nn.Module):
	def __init__(self, in_feats, patch_size, stride, use_zero_init=True):
		"""
		Note: decoupling on channel-dim !
  		"""
		super().__init__()
		self.stride = stride
		self.channel = in_feats
		# print(f"in_feats.shape: {in_feats.shape}")
		# print(f"patch_size.shape: {in_feats.shape}")
		self.patch_size = patch_size

		self.offset_predictor = nn.Sequential(
			nn.Conv1d(1, 64, patch_size, stride=stride, padding=0), 
			nn.GELU(),
			nn.Conv1d(64, 2, 1, 1, padding=0) 
		)

		if use_zero_init:
			self.offset_predictor.apply(zero_init)
		
	def forward(self, X): # Input: (bs, channel, seq_len)
		# (B, C, L)->(B, patch_count, C, 2)
		patch_X = X.unsqueeze(1).permute(0, 1, 3, 2) #128 1 215 1
		# print(f"self.stride: {self.stride}")
		# print(f"input patch_X.shape: {patch_X.shape}")
		patch_X = F.unfold(patch_X, kernel_size=(self.patch_size, self.channel), stride=self.stride).permute(0, 2, 1) # (B, patch_count, patch_size*channel)
		# print(f"patch_X.shape: {patch_X.shape}") #8 12 32
		# decoupling
		B, patch_count = patch_X.shape[0], patch_X.shape[1] 
		patch_X = patch_X.contiguous().view(B, patch_count, self.patch_size, self.channel)
		patch_X = patch_X.permute(0, 1, 3, 2)

		# patch_X: (B, patch_count, channel, patchsize)
		patch_X = patch_X.contiguous().view(B*patch_count*self.channel, 1, self.patch_size)

		# calculate the bias throughout 2 Conv1d
		pred_offset = self.offset_predictor(patch_X)
		pred_offset = pred_offset.view(B, patch_count, self.channel, 2).contiguous()

		# For each of the patch block and it's channel, there exists a bias（dx, ds）
		# pred_offset: (B, patch_count, channel, 2)
		return pred_offset 

# Input: (B, C, L)  Output: (B, C, patch_num * patch_len)
class DepatchSampling(nn.Module):
	def __init__(self, in_feats, seq_len, patch_size, stride):	 
		super(DepatchSampling, self).__init__()
		self.in_feats = in_feats
		self.seq_len = seq_len
		self.patch_size = patch_size

		self.patch_count = (seq_len - patch_size) // stride + 1
		# print(f"patch_count: {self.patch_count}")
		# print(f"self.patch_count in de sampling{self.patch_count}")
		self.patch_count = int(self.patch_count)
		self.dropout = nn.Dropout(0.1)
  
		# offset predictor
		self.offset_predictor = OffsetPredictor(in_feats, patch_size, stride)

		self.box_coder = BoxCoder(self.patch_count, stride, patch_size, self.seq_len, in_feats)
  
	def get_sampling_location(self, X): # Input: (bs, channel, window)  #偏移量计算
		"""
		Input shape: (bs, channel, window) ;
		Sampling location  shape: [bs, patch_count, C, self.patch_size, 2]. range = [0, 1] ; 
		"""
		# get offset (B, C, L)->(B, patch_count, C, 2)  128 12 1 2
		pred_offset = self.offset_predictor(X)
		# print(f"[get sampling]pred_offset: {pred_offset.shape}")
		#(B, patch_count, C, 2) -> (B, patch_count, C, patch_size, 2)
		sampling_locations, bound = self.box_coder(pred_offset)
		# print(f"[get sampling]sampling_locations: {sampling_locations.shape}")

  
  
		return sampling_locations, bound
	
	def forward(self, X, return_bound=False): # Input: (bs, channel, window)
		# Consider the X as a img. shape: (B, C, H, W) <--> (bs, 1, channel, padded_window)  128 1 1 125
		img = X.unsqueeze(1)
		B = img.shape[0]
		# print(f"[DE]After unsqueeze(img): {img.shape}")  # (B, 1, C, L)
  
		#input -> get_sampling_location(input-> OffsetPredictor → BoxCoder) → grid_sample -> output
		#(B, C, L)-> {(B, C, L)->(B, patch_count, C, 2)->(B, patch_count, C, patch_size, 2)}->
		sampling_locations, bound = self.get_sampling_location(X) # sampling_locations: [bs, patch_count, channel, patch_size, 2]
		# print(f"[DE]sampling_locations(before view): {sampling_locations.shape}")  # (B, patch_count, C, patch_size, 2)

		sampling_locations = sampling_locations.view(B, self.patch_count*self.in_feats, self.patch_size, 2)
  
		# print(f"[DE]sampling_locations(after view): {sampling_locations.shape}")  # (B, patch_count, C, patch_size, 2)
		# print(f"[DE]bound shape: {bound.shape}")  # (B, patch_count, 2)
  
  
		# print('sampling_locations: ', sampling_locations.shape)
		#img: 预处理后的输入 (B, 1, C, L);sampling_locations: 采样网格 (B, patch_count*C, patch_size, 2) -> (B, 1, patch_count*C, patch_size)
		sampling_locations = (sampling_locations - 0.5) * 2 # location map: [-1, 1]
		output = F.grid_sample(img, sampling_locations, align_corners=True) 
		# print(f"[DE]grid_sample output shape: {output.shape}")  # (B, 1, patch_count*C, patch_size)
    
  
		#(B, C, patch_count, patch_size)
		output = output.view(B, self.patch_count, self.in_feats, self.patch_size)
		# print(f"output(after view): {output.shape}")  # (B, patch_count, C, patch_size)
    
		output = output.permute(0, 2, 1, 3).contiguous()
		# print(f"Final output shape: {output.shape}")  # (B, C, patch_count, patch_size)

		return output # (B, C, patch_count, patch_size)




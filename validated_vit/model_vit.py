import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# import cv2

MIN_NUM_PATCHES = 16

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # print("-----------in FeedForward, self.net=",self.net)
        # print("-----------in FeedForward, self.net0=",self.net[0])
    def forward(self, x):
        # print("------net[0]=",self.net[0](x))
        return self.net(x)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim/heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1,attention_dropout_rate=0.1,deterministic=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.deterministic = deterministic
        self.input_shape = input_shape
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        # self.layer_norm_input = nn.GroupNorm(1)
        # self.layer_norm_out = nn.GroupNorm(1)

        self.attention = MultiHeadDotProductAttention(input_shape, heads = heads)
        # print("-----------self.attention =",self.attention )
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention  = nn.Dropout(attention_dropout_rate)
    
    def forward(self, inputs):
        x = self.layer_norm_input(inputs)
        x = self.attention(x)
        x = self.drop_out_attention(x)
        x = x + inputs
        y = self.layer_norm_out(x)
        y = self.mlp(y)
        return x + y

class Encoder(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, inputs_positions= None, dropout_rate=0.1, train=False):
        super().__init__()
        self.num_layers = num_layers 
        self.mlp_dim  = mlp_dim
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate
        self.train_flag  = train
        self.encoder_norm = nn.LayerNorm(input_shape)
        # self.encoder_norm = nn.GroupNorm(1)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Encoder1DBlock(input_shape,heads, mlp_dim)]))

    def forward(self, img, mask = None):
        x = img
        for layer in self.layers:
            x = layer[0](x)
        return self.encoder_norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        # how many patches are included in one image, is equal to 196
        num_patches = (image_size // patch_size) ** 2
        # how many pixels are included in one patch, is equal to 768      
        hidden_size = channels * patch_size ** 2           
        print("-----------num_patches=",num_patches)
        print("-----------hidden_size=",hidden_size)
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Conv2d(channels,hidden_size, patch_size, patch_size)  
        # 实现将一张图片进行切分的功能，因为kernel_size=patch_size, stride=patch_size,
        print("-------shape of self_embedding,self_embedding_weight.shape=",self.embedding.weight.shape)  
        # torch.Size([768, 3, 16, 16])
        print("-------shape of self_embedding,self_embedding_bias.shape=",self.embedding.bias.shape)    
        # torch.Size([768])
        # 二维卷积语法介绍：Conv2d(in_channels, out_channels, kernel_size, stride=1)
        # in_channels：输入的通道数目 【必选】
        # out_channels： 输出的通道数目 【必选】
        # kernel_size：卷积核的大小，类型为int 或者元组，当卷积是方形的时候，只需要一个整数边长即可，卷积不是方形，要输入一个元组表示 高和宽。【必选】
        # stride： 卷积每次滑动的步长为多少，默认是 1 【可选】
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        print("-----------self.pos_embedding.shape=",self.pos_embedding.shape)
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Encoder(hidden_size, depth, heads, mlp_dim, dropout_rate = dropout)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Linear(hidden_size, num_classes)

    def forward(self, img, mask = None):
        x = self.embedding(img)
        # 上述x的shape为768*14*14，作用为将图像的每个像素对应的每个RGB分量都通过二维卷积映射为需要的值，但是数量不变
        print("------------x_shape=",x.shape)
        print("------------x_00=",x[0][0:4,0:2,0:3])
        x = rearrange(x, 'b c h w  -> b (h w) c')
        # 上述rearrange的作用为将1*768*14*14的张量映射为1*196*768的张量，即实现张量维度的变换。与维度对应的数据也做相应的变换。
        # 以下的x_00的值与上面的x_00的值可以对应起来，反应出张量数据变换与维度变换的对应关系。
        # 这里解释一下这个结果[1,196,768]是怎么来的。我们知道原始图片向量x的大小为[1,3,224,224]，当我们使用16x16大小的patch对其进行分割的时候，
        #一共可以划分为224x224/16/16 = 196个patches，其次每个patch大小为16x16x3=768，故大小为[1,196,768]。
        # 实际查看原作者的代码，他并没有使用线性映射层来做这件事，出于效率考虑，作者使用了Conv2d层来实现相同的功能。
        # 这是通过设置卷积核大小和步长均为patch_size来实现的。直观上来看，卷积操作是分别应用在每个patch上的。所以，我们可以先应用一个卷积层，然后再对结果进行铺平，
        print("-------------x2_shape=",x.shape)
        print("------------x_00=",x[0][0:3,0:4])
        print("------------x_00=",x[0][14:14+3,0:4])
        b, n, _ = x.shape
        print("---b=",b,"---n=",n)

        print("-----self_cls_shape=",self.cls.shape)
        cls_tokens = repeat(self.cls, '() n d -> b n d', b = b)  #此处cls_tokens=self.cls
        print("----cls_tokens=",cls_tokens.shape)
        # print("------self_cls=",self.cls[0,0,0:10])
        # print("------cls_tokens=",cls_tokens[0,0,0:10])
        #CLS Token
        #下一步是对映射后的patches添加上cls token以及位置编码信息。
        #cls token是一个随机初始化的torch Parameter对象，在forward方法中它需要被拷贝b次(b是batch的数量)，
        #然后使用torch.cat函数添加到patch前面。
        x = torch.cat((cls_tokens, x), dim=1)
        print("----x_shape3=",x.shape)
        print("-----pos_embedding=",self.pos_embedding)
        #Position Embedding
        #目前为止，模型还对patches的在图像中的原始位置一无所知。我们需要传递给模型这些空间上的信息。
        #可以有很多种方法来实现这个功能，在ViT中，我们让模型自己去学习这个。
        #位置编码信息只是一个形状为[N_PATCHES+1(token)m EMBED_SIZE]的张量，它直接加到映射后的patches上。
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

def VIT_B16_224(**kwargs):
    input_size = 224
    patch_size = 16
    num_layers = 12
    num_classes = 2
    if 'num_classes' in kwargs:
        num_classes = kwargs['num_classes']
    print("----------VIT_B16_224----------------")
    return ViT(
        image_size = input_size,
        patch_size = patch_size,
        num_classes = num_classes,
        depth = num_layers,
        heads = 12,
        mlp_dim = 3072,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    
if __name__ == '__main__':
    input_size = 224
    v = VIT_B16_224()
    # v.load_state_dict(torch.load('imagenet21k+imagenet2012_ViT-B_16-224.pth'))
    v.load_state_dict({k.replace('module.',''):v for k,v in torch.load('imagenet21k+imagenet2012_ViT-B_16-224.pth').items()})
    transform_valid = transforms.Compose([
       transforms.Resize((224, 224), interpolation=2),
       transforms.ToTensor()])
    img = Image.open('cat.1.jpg')
    img = Image.open('tcat2.jpg')
    img = Image.open('tdog1.jpg')
    img = Image.open('imgnew.jpg')
    img_ = transform_valid(img).unsqueeze(0) #拓展维度
    print(img_.shape)
    preds = v(img_) # (1, 1000)
    # print(preds)
    kname, indices = torch.max(preds,1)
    print("----kind=",indices)
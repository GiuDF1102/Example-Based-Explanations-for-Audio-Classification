import torch
import torch.nn as nn
from einops import rearrange
from functools import partial

from baselines.ViT.layer_helpers import to_2tuple
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.helpers import load_pretrained

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

class PatchEmbed(nn.Module):
    """Split image into patched and then embed them.
    
    Parameters
    ----------
    img_size : int
        Size of the image (Should be a square).
    
    patch_size : int
        Size of the patch (Should be a square as well).

    in_chans : int
        Number of input channels. (RGB = 3)
    
    embed_dim : int
        The embedding dimension. It is going to stay constant
        across the entire network.

    Attributes
    ----------
    n_patches : int
        Number of patches inside of the image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding. Kernel size and stride are both equal to 
        the patch size, this way when we are sliding the kernel along 
        the input tensor we will never slide it in an overlapping way
        and the kernel will exactly fall into these patches that we 
        are trying to divide our image into.
    """
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`. The n_samples are the batch size, this
            is a batch of images. The image sizes are the height and width.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(x)        # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)        # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (n_samples, n_patches, embed_dim)

        return x
    
class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and output dimension of per token features.
    
    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors. Agaisnt overfitting.

    proj_p : float
        Dropout probability applied to the output tensor. Agaisnt overfitting.
    
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        # Once we concatenate all the attention heads we will get a new tensor that will have
        # The same dimensionality as the input.

        self.scale = head_dim ** -0.5
        # This scaling comes form the AIAYN paper, it prevents feeding extremely large values 
        # into the softmax which could lead to small gradients.

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None # ADDITION TO VIT
        self.attention_map = None  # ADDITION TO VIT
    
    def save_attn_gradients(self, attn_gradients): # ADDITION TO VIT
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self): # ADDITION TO VIT
        return self.attn_gradients

    def save_attention_map(self, attention_map): # ADDITION TO VIT
        self.attention_map = attention_map

    def get_attention_map(self): # ADDITION TO VIT
        return self.attention_map

    def forward(self, x, register_hook = False):
        """Run forward pass. Input and Output are going to have the same
           shape. The second dimension is going to have a + 1 and the 
           reason is that we will have the class token as the first token
           in the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """

        b, n, _, h = *x.shape, self.num_heads

        # self.save_output(x)
        # x.register_hook(self.save_output_grad)

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim) 
        # Take input tensor and turn it into query, keys and values. 

        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) 
        # Permutes and extracts query, keys and values

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        # Dot products

        attn = dots.softmax(dim = -1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn) 
        # Now this is a discrete probability distribution that sums up to 1 and can be used 
        # as weights in a weighted average.

        out = torch.einsum('bhij,bhjd->bhid', attn, v) 
        # This is the weighted average between the attention weights and the values.

        self.save_attention_map(attn)

        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        out = rearrange(out, 'b h n d -> b n (h d)') # (n_samples, n_patches + 1, dim)
        out = self.proj(out) 
        out = self.proj_drop(out) 

        return out
    
class Mlp(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function. 

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------
    dim : int
        Embeddinig dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, num_heads, mlp_ratio = 4., qkv_bias = False, drop = 0., attn_drop = 0., act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads = num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

    def forward(self, x, register_hook = False):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x), register_hook = register_hook) # RESIDUAL BLOCK
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    """Implementation of the Vision transformer.

    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of Transformer blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements. Basically CLS.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements. Includes the information about
        where the patch is in the image.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, num_classes = 1000, embed_dim = 768, depth = 12,
                 num_heads = 12, mlp_ratio = 4., qkv_bias = False, drop_rate = 0., attn_drop_rate = 0., norm_layer = nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        # num_features for consistency with other models
        
        self.patch_embed = PatchEmbed(img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) 
        # Class token parameter initialized with zeros.
         
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # Positional embedding (+1 for class token)

        self.pos_drop = nn.Dropout(p = drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias,
                    drop = drop_rate, attn_drop = attn_drop_rate, norm_layer = norm_layer
                )
                for i in range(depth)
            ]
        )
        # Iterative creation of transformer encoder.

        self.norm = norm_layer(embed_dim)

        
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # Classifier head
        
        trunc_normal_(self.pos_embed, std = .02)
        trunc_normal_(self.cls_token, std = .02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_hook = False):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        B = x.shape[0] 
        # Batch size
        
        x = self.patch_embed(x)
        # Turning into patch embeddings

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # take class token and replicate it over the sample dimension

        x = torch.cat((cls_tokens, x), dim=1)
        # Prepend class token to patch embeddings.

        x = x + self.pos_embed
        # Add positional embeddings

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, register_hook=register_hook)
        # Iteratively define all the blocks of the transformer encoder.

        x = self.norm(x)
        
        x = x[:, 0]
        x = self.head(x)
        # Select only the CLASS embedding which is brought to the classifier
        
        return x 
    
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
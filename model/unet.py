import torch
import torch.nn as nn
from utils.sinusoidal_embeddings import get_sinusoidal_embeddings
from model.unet_utils import UnetUtils

class DownBlock(nn.Module):
    """
        DownBlock part of the Unet.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, att_head_num: int=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.att_head_num = att_head_num

        self.resnet_conv1, self.resnet_conv2 = UnetUtils.resnet1(
            in_channels=in_channels,
            out_channels=out_channels,
            block="down"
        )

        self.time_emb_layer1, self.time_emb_layer2 = UnetUtils.temb_layer(
            time_emb_dim=time_emb_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            block="down"
        )

        self.resnet_conv3, self.resnet_conv4 = UnetUtils.resnet2(
            in_channels=in_channels,
            out_channels=out_channels,
            block="down"
        )

        self.attention_norms1, self.attentions1, self.attention_norms2, self.attentions2 = UnetUtils.attentions(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=8,
            att_head_num=att_head_num,
            block="down"
        )
        self.down_sample_conv = UnetUtils.down_sample(out_channels=out_channels)

    def forward(self, x, time_embs):
        """
            Forward function for the DownBlock
        """
        out = self.resnet_conv1(x)
        out = out + self.time_emb_layer1(time_embs)[:, :, None, None]
        out = self.resnet_conv2(out)

        batch_size, channels_num , h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms1(input_att)
        input_att = input_att.transpose(1, 2)

        out_att, _ = self.attentions1(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att
        out1 = out 

        out = self.resnet_conv3(out)
        out = out + self.time_emb_layer2(time_embs)[:, :, None, None]
        out = self.resnet_conv4(out)

        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms2(input_att)
        input_att = input_att.transpose(1, 2)
        out_att, _ = self.attentions2(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att
        out2 = out

        out_down = self.down_sample_conv(out)

        return [out1, out2], out_down


class MidBlock(nn.Module):
    """
        MidBlock part of the Unet.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, att_head_num=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.att_head_num = att_head_num

        self.resnet_conv1, self.resnet_conv2 = UnetUtils.resnet1(
            in_channels=in_channels,
            out_channels=out_channels,
            block="mid"
        )
        self.time_emb_layer1, self.time_emb_layer2 = UnetUtils.temb_layer(
            time_emb_dim=time_emb_dim,
            in_channels=in_channels, 
            out_channels=out_channels,
            block="mid"
        )
        self.resnet_conv3, self.resnet_conv4 = UnetUtils.resnet2(
            in_channels=in_channels,
            out_channels=out_channels,
            block="mid"
        )
        self.attention_norms, self.attentions = UnetUtils.attentions(
            in_channels=in_channels,
            out_channels=out_channels, 
            num_groups=8,
            att_head_num=att_head_num,
            block="mid"
        )
        

    def forward(self, x, time_embs):
        # iter 1 
        out = self.resnet_conv1(x)
        out = out + self.time_emb_layer1(time_embs)[:, :, None, None]
        out = self.resnet_conv2(out)

        batch_size, channels_num, h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms(input_att)
        input_att = input_att.transpose(1, 2)

        out_att, _ = self.attentions(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att

        # iter 2
        out = self.resnet_conv3(x)
        out = out + self.time_emb_layer2(time_embs)[:, :, None, None]
        out = self.resnet_conv4(out)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, att_head_num=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.att_head_num = att_head_num

        self.resnet_conv1, self.resnet_conv2 = UnetUtils.resnet1(
            in_channels=in_channels,
            out_channels=out_channels,
            block="up"
        )
        self.time_emb_layer1, self.time_emb_layer2 = UnetUtils.temb_layer(
            time_emb_dim=time_emb_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            block="up"
        )

        self.resnet_conv3, self.resnet_conv4 = UnetUtils.resnet2(
            in_channels=in_channels,
            out_channels=out_channels,
            block="up"
        )

        self.attention_norms1, self.attentions1, self.attention_norms2, self.attentions2 = UnetUtils.attentions(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=8,
            att_head_num=att_head_num,
            block="up"
        )
        self.up_sample_conv = UnetUtils.up_sample(in_channels=in_channels)

    def forward(self, x, out_lst, time_embs):
        """
            Forward function for the UpBlock.
        """
        resnet_input = x
        x = self.up_sample_conv(x)
        x1 = torch.cat([x, out_lst[1]], dim=1)

        out = self.resnet_conv1(x1)
        out = out + self.time_emb_layer1(time_embs)[:, :, None, None]
        out = self.resnet_conv2(out)
        #out = out + self.residual_input_conv(resnet_input)

        batch_size, channels_num , h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms1(input_att)
        input_att = input_att.transpose(1, 2)

        out_att, _ = self.attentions1(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att

        x2 = torch.cat([out, out_lst[0]], dim=1)
        #print(out.shape, out_lst[0].shape)
        out = self.resnet_conv3(x2)
        out = out + self.time_emb_layer2(time_embs)[:, :, None, None]
        out = self.resnet_conv4(out)

        batch_size, channels_num, h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms2(input_att)
        input_att = input_att.transpose(1, 2)
        out_att, _ = self.attentions2(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att


        return out
    

class Unet(nn.Module):
    """
        Main Unet class, which assembles the down, mid, and up block and runs forward function.
    """
    def __init__(self, channels_lst=[32, 64], im_channels=1, time_emb_dim=256):
        super().__init__()

        self.channels_lst = channels_lst
        self.im_channels = im_channels
        self.time_emb_dim = time_emb_dim

        self.down_blocks = []
        for channel in channels_lst:
            self.down_blocks.append(DownBlock(in_channels=channel, out_channels=channel*2, time_emb_dim=256))

        mid_block_channels_num = channels_lst[-1]*2
        self.mid_block = MidBlock(in_channels=mid_block_channels_num, out_channels=mid_block_channels_num, time_emb_dim=256)

        self.up_blocks = []
        for channel in list(reversed(channels_lst)):
            self.up_blocks.append(UpBlock(in_channels=channel * 2 , out_channels=channel, time_emb_dim=256))

        self._init_conv_in()
        self._init_conv_out()
        self._init_time_proj()


    def _init_conv_in(self):
        self.conv_in = nn.Conv2d(self.im_channels, self.channels_lst[0], kernel_size=3, padding=(1, 1))

    def _init_conv_out(self, group_num=8):
        self.norm_out = nn.GroupNorm(group_num, self.channels_lst[0])
        self.conv_out = nn.Conv2d(self.channels_lst[0], self.im_channels, kernel_size=3, padding=(1, 1))
    
    def _init_time_proj(self):
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

    def forward(self, x, timesteps):
        out = self.conv_in(x)

        time_embs = get_sinusoidal_embeddings(torch.as_tensor(timesteps).long(), self.time_emb_dim)
        time_embs = self.time_proj(time_embs)

       
        down_out_lst = []
        for down in self.down_blocks:
            out_lst, out = down(out, time_embs)
            down_out_lst.append(out_lst)

        out = self.mid_block(out, time_embs) 

        rev_down_outs = list(reversed(down_out_lst))
        for i, up in enumerate(self.up_blocks):
            out_lst = rev_down_outs[i]
            out = up(out, out_lst, time_embs)
        
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out



        






        

        






